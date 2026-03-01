"""Greedy agent that evaluates each action with a simple heuristic."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

from .base import Agent

if TYPE_CHECKING:
    from ..core.actions import Action
    from ..core.game import GameConfig
    from ..core.game_state import GameState
    from ..core.types import TeamId


class GreedyAgent(Agent):
    """Agent that greedily picks the highest-scoring action.

    Scoring heuristic:
    - +10000 if action completes a sequence
    - +1000 if extends a line to 4 of own chips
    - +500 if blocks opponent's line of 4
    - +100 if extends a line to 3 of own chips
    - +50 for center-ish positions
    - +10 if extends a line to 2 of own chips

    Tiebreak: random.
    """

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)
        self._team: TeamId | None = None
        self._num_teams: int = 2

    def notify_game_start(self, team: TeamId, config: GameConfig) -> None:
        self._team = team
        self._num_teams = config.num_teams

    def choose_action(self, state: GameState, legal_actions: list[Action]) -> Action:
        from ..core.actions import ActionType
        from ..core.board import ALL_LINES, POSITION_TO_LINES
        from ..core.types import CORNER, EMPTY, TeamId

        team = self._team
        if team is None:
            team = state.current_team

        scored: list[tuple[float, float, Action]] = []

        for action in legal_actions:
            score = 0.0

            if action.action_type == ActionType.DEAD_CARD_DISCARD:
                # Dead card discard: minimal score, just get rid of it
                score = -1.0
                scored.append((score, self._rng.random(), action))
                continue

            pos = action.position
            assert pos is not None

            if action.action_type == ActionType.PLACE:
                # Simulate the placement
                new_state = state.copy()
                new_state.board.place_chip(pos, team)
                new_seqs = new_state.board.check_new_sequences(pos, team)

                if new_seqs:
                    score += 10000 * len(new_seqs)
                else:
                    # Evaluate how this placement improves our lines
                    line_indices = POSITION_TO_LINES.get(pos, [])
                    for line_idx in line_indices:
                        line = ALL_LINES[line_idx]
                        own = 0
                        empty = 0
                        opponent = 0
                        for p in line:
                            chip = new_state.board.get_chip(p)
                            if chip == team.value or chip == CORNER:
                                own += 1
                            elif chip == EMPTY:
                                empty += 1
                            else:
                                opponent += 1
                        # Only count lines that are still viable (no opponent chips)
                        if opponent == 0:
                            if own == 4:
                                score += 1000
                            elif own == 3:
                                score += 100
                            elif own == 2:
                                score += 10

                    # Check if blocking opponent lines of 4
                    for opp_val in range(self._num_teams):
                        if opp_val == team.value:
                            continue
                        opp_team = TeamId(opp_val)
                        line_indices_for_pos = POSITION_TO_LINES.get(pos, [])
                        for line_idx in line_indices_for_pos:
                            line = ALL_LINES[line_idx]
                            # Count opponent chips in original state (before our placement)
                            opp_count = 0
                            for p in line:
                                chip = state.board.get_chip(p)
                                if chip == opp_val or chip == CORNER:
                                    opp_count += 1
                            # If opponent had 4 of 5 and we just placed on the 5th
                            if opp_count == 4:
                                score += 500

                    # Center position bonus
                    center_dist = abs(pos.row - 4.5) + abs(pos.col - 4.5)
                    if center_dist <= 3:
                        score += 50

            elif action.action_type == ActionType.REMOVE:
                # Removing an opponent chip: evaluate blocking value
                for opp_val in range(self._num_teams):
                    if opp_val == team.value:
                        continue
                    line_indices = POSITION_TO_LINES.get(pos, [])
                    for line_idx in line_indices:
                        line = ALL_LINES[line_idx]
                        opp_count = 0
                        for p in line:
                            chip = state.board.get_chip(p)
                            if chip == opp_val or chip == CORNER:
                                opp_count += 1
                        if opp_count >= 4:
                            score += 500
                        elif opp_count >= 3:
                            score += 100

            scored.append((score, self._rng.random(), action))

        # Sort by score descending, then random tiebreak
        scored.sort(key=lambda x: (-x[0], x[1]))
        return scored[0][2]
