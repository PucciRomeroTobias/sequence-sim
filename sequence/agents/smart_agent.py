"""Smart agent combining card counting, enhanced features, and optional lookahead."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import Agent
from ..core.card_tracker import CardTracker
from ..scoring.scoring_function import SMART_WEIGHTS, ScoringFunction, ScoringWeights

if TYPE_CHECKING:
    from ..core.actions import Action
    from ..core.game import GameConfig
    from ..core.game_state import GameState
    from ..core.types import TeamId


class SmartAgent(Agent):
    """Agent that uses card counting + 33-feature evaluation + instant decisions.

    Combines:
    - CardTracker for perfect public information tracking
    - Enhanced 33-feature scoring with card-availability awareness
    - Instant decisions for obvious moves (complete sequence, block 4-in-a-row)
    - Jack timing: preserves jacks for critical moments
    - Optional depth-1 lookahead on top candidates
    """

    def __init__(
        self,
        weights: ScoringWeights | None = None,
        use_lookahead: bool = True,
        lookahead_candidates: int = 5,
    ) -> None:
        self._weights = weights or SMART_WEIGHTS
        self._scoring = ScoringFunction(self._weights)
        self._use_lookahead = use_lookahead
        self._lookahead_candidates = lookahead_candidates
        self._tracker: CardTracker | None = None
        self._team: TeamId | None = None

    def notify_game_start(self, team: TeamId, config: GameConfig) -> None:
        from ..core.types import TeamId as TId
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

        # --- Instant decisions ---
        instant = self._check_instant_decisions(state, legal_actions, team)
        if instant is not None:
            return instant

        # --- Score all actions (fast virtual apply for PLACE actions) ---
        scored = self._scoring.rank_actions_fast(
            state, legal_actions, team, tracker=self._tracker
        )

        # --- Dead card boost: incentivize cycling dead cards when close to best ---
        if len(scored) > 1:
            from ..core.actions import ActionType as AT
            best_score = scored[0][1]
            boosted = False
            for i, (action, score) in enumerate(scored):
                if (
                    action.action_type == AT.DEAD_CARD_DISCARD
                    and best_score - score <= 3.0
                ):
                    scored[i] = (action, score + 2.0)
                    boosted = True
            if boosted:
                scored.sort(key=lambda x: x[1], reverse=True)

        if not self._use_lookahead or len(scored) <= 1:
            return scored[0][0]

        # --- Depth-1 lookahead on top candidates ---
        top = scored[: self._lookahead_candidates]
        best_action = top[0][0]
        best_value = top[0][1]

        for action, base_score in top:
            next_state = state.apply_action(action)
            # Check if this wins
            winner = next_state.is_terminal()
            if winner == team:
                return action

            # Evaluate opponent's best response
            opp_team = next_state.current_team
            opp_actions = next_state.get_legal_actions(opp_team)
            if not opp_actions:
                value = base_score
            else:
                # Find opponent's best move
                worst_for_us = float("inf")
                for opp_action in opp_actions[:10]:  # Limit for speed
                    after_opp = next_state.apply_action(opp_action)
                    our_score = self._scoring.evaluate(
                        after_opp, team, tracker=self._tracker
                    )
                    if our_score < worst_for_us:
                        worst_for_us = our_score
                value = worst_for_us

            if value > best_value:
                best_value = value
                best_action = action

        return best_action

    def _check_instant_decisions(
        self,
        state: GameState,
        legal_actions: list[Action],
        team: TeamId,
    ) -> Action | None:
        """Check for obvious best moves that don't need scoring."""
        from ..core.actions import ActionType

        board = state.board

        # 1. Complete a sequence if possible
        for action in legal_actions:
            if action.action_type == ActionType.PLACE and action.position:
                next_state = state.apply_action(action)
                if next_state.board.count_sequences(team) > board.count_sequences(team):
                    return action

        # 2. Block opponent 4-in-a-row
        from ..core.board import ALL_LINES, POSITION_TO_LINES
        from ..core.types import CORNER, CORNERS, EMPTY, Position

        chips = board.chips
        opp_vals: set[int] = set()
        for t in range(state.num_teams):
            if t != team.value:
                opp_vals.add(t)

        # Find positions that block opponent's 4-in-a-row
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
            # Collect all blocking actions, preferring normal cards over jacks
            jack_block: Action | None = None
            for action in legal_actions:
                if (
                    action.action_type == ActionType.PLACE
                    and action.position in blocking_positions
                ):
                    if action.card.is_two_eyed_jack:
                        # Remember jack option but keep looking for normal card
                        if jack_block is None:
                            jack_block = action
                    else:
                        # Normal card block — preferred, preserves jacks
                        return action
            # Fall back to jack block if no normal card can block
            if jack_block is not None:
                return jack_block

        # 3. Fork creation: position advancing 2+ lines with 3+ own chips each
        fork_positions: set = set()
        for r in range(10):
            for c in range(10):
                pos = Position(r, c)
                if int(chips[pos.row, pos.col]) != EMPTY:
                    continue
                if pos in CORNERS:
                    continue
                lines_advanced = 0
                line_indices = POSITION_TO_LINES.get(pos, [])
                for line_idx in line_indices:
                    line = ALL_LINES[line_idx]
                    own_in_line = 0
                    opp_in_line = 0
                    for lp in line:
                        v = int(chips[lp.row, lp.col])
                        if v == team.value or v == CORNER:
                            own_in_line += 1
                        elif v in opp_vals:
                            opp_in_line += 1
                    if opp_in_line == 0 and own_in_line >= 3:
                        lines_advanced += 1
                if lines_advanced >= 2:
                    fork_positions.add(pos)

        if fork_positions:
            jack_fork: Action | None = None
            for action in legal_actions:
                if (
                    action.action_type == ActionType.PLACE
                    and action.position in fork_positions
                ):
                    if action.card.is_two_eyed_jack:
                        if jack_fork is None:
                            jack_fork = action
                    else:
                        return action
            if jack_fork is not None:
                return jack_fork

        return None
