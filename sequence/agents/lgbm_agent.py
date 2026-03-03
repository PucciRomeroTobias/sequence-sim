"""LightGBM agents: MCTS-distilled action ranking.

Two agent variants:
- LGBMAgent: Pure ranking, no lookahead (fast but weaker)
- HybridAgent: LGBM candidate selection + linear lookahead (best overall)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import Agent
from ..core.card_tracker import CardTracker

if TYPE_CHECKING:
    from ..core.actions import Action
    from ..core.game import GameConfig
    from ..core.game_state import GameState
    from ..core.types import TeamId


class LGBMAgent(Agent):
    """Agent using LightGBM LambdaRank for action selection.

    Pure ranking model — no lookahead. Fast but weaker than HybridAgent.
    """

    def __init__(self, model_path: str) -> None:
        from ..scoring.lgbm_scoring import LGBMScoringFunction

        self._scoring = LGBMScoringFunction(model_path)
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

        if len(legal_actions) == 1:
            return legal_actions[0]

        instant = _check_instant_decisions(state, legal_actions, team)
        if instant is not None:
            return instant

        scored = self._scoring.rank_actions(
            state, legal_actions, team, tracker=self._tracker
        )
        return scored[0][0]


class HybridAgent(Agent):
    """Best agent: LGBM candidate selection + SmartAgent-style linear lookahead.

    LGBM (trained on MCTS visits) provides better candidate ranking than
    linear scoring (48.7% vs 25.4% MCTS top-1 agreement). Linear lookahead
    then verifies candidates tactically with calibrated cross-state scores.
    """

    def __init__(
        self,
        model_path: str,
        lookahead_candidates: int = 5,
    ) -> None:
        from ..scoring.lgbm_scoring import LGBMScoringFunction
        from ..scoring.scoring_function import ScoringFunction, SMART_WEIGHTS

        self._lgbm = LGBMScoringFunction(model_path)
        self._linear = ScoringFunction(SMART_WEIGHTS)
        self._top_k = lookahead_candidates
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

        if len(legal_actions) == 1:
            return legal_actions[0]

        # Instant decisions (always correct, skip ML)
        instant = _check_instant_decisions(state, legal_actions, team)
        if instant is not None:
            return instant

        # LGBM ranks all actions, take top-K candidates
        scored = self._lgbm.rank_actions(
            state, legal_actions, team, tracker=self._tracker
        )
        top = scored[: self._top_k]

        if len(top) <= 1:
            return top[0][0]

        # Linear depth-1 lookahead on candidates
        best_action = top[0][0]
        best_value = float("-inf")

        for action, _ in top:
            next_state = state.apply_action(action)
            winner = next_state.is_terminal()
            if winner == team:
                return action

            opp_team = next_state.current_team
            opp_actions = next_state.get_legal_actions(opp_team)
            if not opp_actions:
                value = self._linear.evaluate(
                    next_state, team, tracker=self._tracker
                )
            else:
                worst_for_us = float("inf")
                for opp_action in opp_actions[:10]:
                    after_opp = next_state.apply_action(opp_action)
                    score = self._linear.evaluate(
                        after_opp, team, tracker=self._tracker
                    )
                    if score < worst_for_us:
                        worst_for_us = score
                value = worst_for_us

            if value > best_value:
                best_value = value
                best_action = action

        return best_action


def _check_instant_decisions(
    state: GameState,
    legal_actions: list[Action],
    team: TeamId,
) -> Action | None:
    """Check for obvious best moves that don't need scoring."""
    from ..core.actions import ActionType
    from ..core.board import ALL_LINES
    from ..core.types import CORNER, EMPTY

    board = state.board
    chips = board.chips

    # 1. Complete a sequence if possible
    for action in legal_actions:
        if action.action_type == ActionType.PLACE and action.position:
            next_state = state.apply_action(action)
            if next_state.board.count_sequences(team) > board.count_sequences(team):
                return action

    # 2. Block opponent 4-in-a-row
    opp_vals: set[int] = set()
    for t in range(state.num_teams):
        if t != team.value:
            opp_vals.add(t)

    blocking_positions: set = set()
    for line in ALL_LINES:
        opp_count = 0
        own_count = 0
        corner_count = 0
        empty_pos = None
        for pos in line:
            val = int(chips[pos.row, pos.col])
            if val in opp_vals:
                opp_count += 1
            elif val == team.value:
                own_count += 1
            elif val == CORNER:
                corner_count += 1
            elif val == EMPTY:
                empty_pos = pos

        opp_total = opp_count + corner_count
        if own_count == 0 and opp_total == 4 and empty_pos is not None:
            blocking_positions.add(empty_pos)

    if blocking_positions:
        jack_block: Action | None = None
        for action in legal_actions:
            if (
                action.action_type == ActionType.PLACE
                and action.position in blocking_positions
            ):
                if action.card.is_two_eyed_jack:
                    if jack_block is None:
                        jack_block = action
                else:
                    return action
        if jack_block is not None:
            return jack_block

    return None
