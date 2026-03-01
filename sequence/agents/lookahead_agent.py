"""Lookahead agent using minimax with alpha-beta pruning."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

from .base import Agent

if TYPE_CHECKING:
    from ..core.actions import Action
    from ..core.game import GameConfig
    from ..core.game_state import GameState
    from ..core.types import TeamId


class LookaheadAgent(Agent):
    """Minimax agent with alpha-beta pruning and configurable depth.

    For depth=1: evaluates own moves.
    For depth=2: evaluates own move + opponent's best response.
    """

    def __init__(
        self,
        depth: int = 1,
        scoring_fn: object | None = None,
        max_actions: int = 15,
        seed: int | None = None,
    ) -> None:
        self._depth = depth
        self._scoring_fn = scoring_fn
        self._max_actions = max_actions
        self._rng = random.Random(seed)
        self._team: TeamId | None = None

    def notify_game_start(self, team: TeamId, config: GameConfig) -> None:
        self._team = team

    def notify_action(self, action: Action, team: TeamId) -> None:
        pass

    def choose_action(
        self, state: GameState, legal_actions: list[Action]
    ) -> Action:
        if self._team is None:
            self._team = state.current_team

        if len(legal_actions) == 1:
            return legal_actions[0]

        # Pre-filter actions if too many (e.g., two-eyed jacks)
        actions = self._filter_actions(state, legal_actions)

        best_score = float("-inf")
        best_actions: list[Action] = []

        for action in actions:
            new_state = state.apply_action(action)
            score = self._minimax(
                new_state,
                self._depth - 1,
                float("-inf"),
                float("inf"),
                False,  # Next is opponent's turn (minimizing)
            )
            if score > best_score:
                best_score = score
                best_actions = [action]
            elif score == best_score:
                best_actions.append(action)

        return self._rng.choice(best_actions)

    def _minimax(
        self,
        state: GameState,
        depth: int,
        alpha: float,
        beta: float,
        maximizing: bool,
    ) -> float:
        # Terminal check
        winner = state.is_terminal()
        if winner is not None:
            if winner == self._team:
                return 100000.0
            else:
                return -100000.0

        if depth == 0:
            return self._evaluate(state)

        legal_actions = state.get_legal_actions()
        if not legal_actions:
            return self._evaluate(state)

        # Filter actions for performance
        actions = self._filter_actions(state, legal_actions)

        if maximizing:
            value = float("-inf")
            for action in actions:
                new_state = state.apply_action(action)
                value = max(
                    value,
                    self._minimax(new_state, depth - 1, alpha, beta, False),
                )
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return value
        else:
            value = float("inf")
            for action in actions:
                new_state = state.apply_action(action)
                value = min(
                    value,
                    self._minimax(new_state, depth - 1, alpha, beta, True),
                )
                beta = min(beta, value)
                if alpha >= beta:
                    break
            return value

    def _filter_actions(
        self, state: GameState, actions: list[Action]
    ) -> list[Action]:
        """Pre-filter to top K actions by quick heuristic when too many."""
        if len(actions) <= self._max_actions:
            return actions

        # Score each action quickly
        scored: list[tuple[float, Action]] = []
        for action in actions:
            score = self._quick_score(state, action)
            scored.append((score, action))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [a for _, a in scored[: self._max_actions]]

    def _quick_score(self, state: GameState, action: Action) -> float:
        """Fast heuristic score for action pre-filtering."""
        from ..core.actions import ActionType
        from ..core.types import Position

        score = 0.0

        if action.action_type == ActionType.PLACE and action.position:
            pos = action.position
            # Center preference
            center_dist = abs(pos.row - 4.5) + abs(pos.col - 4.5)
            score += max(0, 5 - center_dist)

            # Check if completes something
            new_state = state.apply_action(action)
            team = state.current_team
            new_seqs = new_state.board.count_sequences(team)
            old_seqs = state.board.count_sequences(team)
            if new_seqs > old_seqs:
                score += 10000

        elif action.action_type == ActionType.REMOVE and action.position:
            # Removing opponent chip is valuable
            score += 50

        return score

    def _evaluate(self, state: GameState) -> float:
        """Evaluate state using scoring function or built-in heuristic."""
        if self._scoring_fn is not None:
            return self._scoring_fn.evaluate(state, self._team)

        # Built-in heuristic
        return self._builtin_evaluate(state)

    def _builtin_evaluate(self, state: GameState) -> float:
        """Simple built-in evaluation function."""
        from ..core.board import ALL_LINES
        from ..core.types import CORNER

        team = self._team
        assert team is not None
        team_val = team.value
        opp_vals = [
            t for t in range(state.num_teams) if t != team_val
        ]

        board = state.board
        chips = board.chips
        score = 0.0

        # Sequence count
        score += board.count_sequences(team) * 10000

        for opp in opp_vals:
            from ..core.types import TeamId as TId

            score -= board.count_sequences(TId(opp)) * 10000

        # Count lines potential
        for line in ALL_LINES:
            own_count = 0
            opp_count = 0
            for pos in line:
                val = int(chips[pos.row, pos.col])
                if val == team_val or val == CORNER:
                    own_count += 1
                elif val != -1 and val != CORNER:
                    opp_count += 1

            if opp_count == 0 and own_count >= 2:
                score += {2: 10, 3: 100, 4: 1000, 5: 0}.get(own_count, 0)
            if own_count == 0 and opp_count >= 2:
                score -= {2: 5, 3: 80, 4: 800, 5: 0}.get(opp_count, 0)

        return score
