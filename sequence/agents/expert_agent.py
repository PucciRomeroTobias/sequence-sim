"""Expert agent using prioritized heuristic cascade with smart lookahead."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .base import Agent
from ..core.card_tracker import CardTracker
from ..scoring.scoring_function import SMART_WEIGHTS, ScoringFunction, ScoringWeights
from .expert.removal import score_removal
from .expert.stance import OPENING, compute_stance
from .expert.tactics import check_instant_decisions

if TYPE_CHECKING:
    from ..core.actions import Action
    from ..core.game import GameConfig
    from ..core.game_state import GameState
    from ..core.types import TeamId


class ExpertAgent(Agent):
    """Expert-level agent with tiered heuristic cascade and smart lookahead.

    Improvements over SmartAgent:
    - Enhanced Tier 1: double-threat detection, opponent fork prevention, force-win
    - Intelligent removal scoring for one-eyed jacks
    - Smarter lookahead: pre-sorts opponent responses by threat level
    - Jack lockout in opening phase
    - Scaled dead card boost
    - Tiebreaking: conserve jacks, maximize optionality

    When custom weights are provided, uses 47-feature expert extraction with
    optional normalization scale factors.
    """

    def __init__(
        self,
        use_lookahead: bool = True,
        lookahead_candidates: int = 7,
        weights: ScoringWeights | None = None,
        scale_factors: np.ndarray | None = None,
    ) -> None:
        if weights is not None:
            self._scoring = ScoringFunction(
                weights,
                use_expert_features=True,
                scale_factors=scale_factors,
            )
        else:
            self._scoring = ScoringFunction(SMART_WEIGHTS)
        self._use_lookahead = use_lookahead
        self._lookahead_candidates = lookahead_candidates
        self._tracker: CardTracker | None = None
        self._team: TeamId | None = None

    def notify_game_start(self, team: TeamId, config: GameConfig) -> None:
        self._team = team
        self._tracker = CardTracker(team, config.num_teams)

    def notify_action(self, action: Action, team: TeamId) -> None:
        if self._tracker is not None:
            self._tracker.on_action(action, team)

    def choose_action(
        self, state: GameState, legal_actions: list[Action]
    ) -> Action:
        if self._team is None:
            self._team = state.current_team
        if self._tracker is None:
            self._tracker = CardTracker(self._team, state.num_teams)

        team = self._team
        hand = state.hands.get(team.value, [])
        self._tracker.sync_hand(hand)

        # --- Tier 0: Trivial ---
        if len(legal_actions) == 1:
            return legal_actions[0]

        # --- Tier 1: Enhanced instant decisions ---
        instant = check_instant_decisions(state, legal_actions, team, self._tracker)
        if instant is not None:
            return instant

        # --- Stance-based filtering ---
        stance = compute_stance(state, team, self._tracker)
        if stance == OPENING:
            non_jack = [a for a in legal_actions if not a.card.is_jack]
            if non_jack:
                legal_actions = non_jack

        # --- Tier 3: Score all actions ---
        scored = self._score_actions(state, legal_actions, team)

        # --- Tier 4: Dead card boost (scaled) ---
        scored = self._boost_dead_cards(scored)

        if not self._use_lookahead or len(scored) <= 1:
            return scored[0][0]

        # --- Tier 5: Smart lookahead ---
        top = scored[: self._lookahead_candidates]
        best = self._smart_lookahead(state, top, team)

        return best

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _score_actions(
        self,
        state: GameState,
        legal_actions: list[Action],
        team: TeamId,
    ) -> list[tuple[Action, float]]:
        """Score actions using SmartAgent's proven scoring + removal bonus."""
        from ..core.actions import ActionType

        # Use the fast virtual-apply scoring from ScoringFunction
        scored = self._scoring.rank_actions_fast(
            state, legal_actions, team, tracker=self._tracker
        )

        # Add scaled removal scoring bonus to REMOVE actions
        # Scale factor keeps bonus proportional (avoids overriding base scoring)
        result: list[tuple[Action, float]] = []
        for action, score in scored:
            if action.action_type == ActionType.REMOVE:
                bonus = score_removal(action, state, team, self._tracker) * 0.15
                result.append((action, score + bonus))
            else:
                result.append((action, score))

        # Sort by score (desc), then tiebreak key (asc) for equal scores
        result.sort(key=lambda x: (-x[1], self._tiebreak_key(x[0])))
        return result

    # ------------------------------------------------------------------
    # Dead card boost
    # ------------------------------------------------------------------

    def _boost_dead_cards(
        self,
        scored: list[tuple[Action, float]],
    ) -> list[tuple[Action, float]]:
        """Boost dead card discards scaled by dead card count in hand."""
        if len(scored) <= 1:
            return scored

        from ..core.actions import ActionType

        dead_count = sum(
            1 for action, _ in scored
            if action.action_type == ActionType.DEAD_CARD_DISCARD
        )
        if dead_count == 0:
            return scored

        # Scale boost: 1 dead=+2, 2 dead=+4, 3+=+6
        boost = min(2.0 * dead_count, 6.0)
        best_score = scored[0][1]

        boosted = False
        result = list(scored)
        for i, (action, score) in enumerate(result):
            if (
                action.action_type == ActionType.DEAD_CARD_DISCARD
                and best_score - score <= 3.0
            ):
                result[i] = (action, score + boost)
                boosted = True

        if boosted:
            result.sort(key=lambda x: x[1], reverse=True)
        return result

    # ------------------------------------------------------------------
    # Smart lookahead
    # ------------------------------------------------------------------

    def _smart_lookahead(
        self,
        state: GameState,
        candidates: list[tuple[Action, float]],
        team: TeamId,
    ) -> Action:
        """Depth-1 lookahead that evaluates the most THREATENING opponent responses.

        Key improvement over SmartAgent: instead of evaluating the first N opponent
        actions (which are often random jack placements), we pre-score opponent actions
        and focus on the ones that hurt us most.
        """
        best_action = candidates[0][0]
        best_value = candidates[0][1]

        for action, base_score in candidates:
            next_state = state.apply_action(action)

            # Check for immediate win
            winner = next_state.is_terminal()
            if winner == team:
                return action

            # Evaluate opponent's best response
            opp_team = next_state.current_team
            opp_actions = next_state.get_legal_actions(opp_team)
            if not opp_actions:
                value = base_score
            else:
                value = self._evaluate_opponent_threats(
                    next_state, opp_actions, team, opp_team
                )

            if value > best_value:
                best_value = value
                best_action = action

        return best_action

    def _evaluate_opponent_threats(
        self,
        state: GameState,
        opp_actions: list[Action],
        our_team: TeamId,
        opp_team: TeamId,
    ) -> float:
        """Evaluate opponent's best responses, prioritizing threatening moves.

        Instead of evaluating the first 10 arbitrary actions, we:
        1. Quick-score all opponent actions from their perspective
        2. Evaluate the top threats from OUR perspective
        This ensures we consider the opponent's strongest responses.
        """
        from ..core.actions import ActionType

        # Quick-score opponent actions from THEIR perspective
        opp_scored = self._scoring.rank_actions_fast(
            state, opp_actions, opp_team, tracker=self._tracker
        )

        # Take top threats (opponent's best moves)
        top_threats = opp_scored[: 10]

        # Evaluate each from our perspective
        worst_for_us = float("inf")
        for opp_action, _opp_score in top_threats:
            after_opp = state.apply_action(opp_action)
            our_score = self._scoring.evaluate(
                after_opp, our_team, tracker=self._tracker
            )
            if our_score < worst_for_us:
                worst_for_us = our_score

        return worst_for_us

    # ------------------------------------------------------------------
    # Tiebreaking
    # ------------------------------------------------------------------

    def _tiebreak_key(self, action: Action) -> tuple:
        """Secondary sort key for tiebreaking equal-scored actions.

        Prefer:
        1. Normal cards over jacks (conserve jacks for critical moments)
        2. Common cards over rare ones (reveal less information)
        """
        card = action.card
        # 1. Jack penalty: prefer normal cards
        jack_penalty = 1 if card.is_jack else 0

        # 2. Rarity — fewer copies remaining in pool = rarer = prefer playing common
        rarity = 0
        if self._tracker is not None and not card.is_jack:
            rarity = -self._tracker.copies_remaining_in_pool(card)

        return (jack_penalty, rarity)
