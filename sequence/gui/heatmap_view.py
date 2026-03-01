"""Heatmap computation for board position scoring."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..core.actions import ActionType
from ..core.types import CORNERS, Position

if TYPE_CHECKING:
    from ..core.game_state import GameState
    from ..core.types import TeamId
    from ..scoring.scoring_function import ScoringFunction


def compute_scoring_heatmap(
    state: GameState,
    team: TeamId,
    scoring_fn: ScoringFunction,
) -> np.ndarray:
    """Compute a 10x10 float array of scores for each board position.

    For each empty position, simulates a PLACE action with a two-eyed jack
    and evaluates the resulting state. Occupied positions and corners get 0.

    The result is normalized to [0, 1] range.

    Returns:
        10x10 numpy array of float values in [0, 1].
    """
    from ..core.actions import Action
    from ..core.card import Card
    from ..core.types import Rank, Suit

    heatmap = np.zeros((10, 10), dtype=np.float64)
    wild_card = Card(Rank.JACK, Suit.DIAMONDS)  # Two-eyed jack for simulation

    scores: list[float] = []
    positions: list[tuple[int, int]] = []

    for r in range(10):
        for c in range(10):
            pos = Position(r, c)
            if pos in CORNERS:
                continue
            if not state.board.is_empty(pos):
                continue

            # Simulate placing a chip here
            action = Action(wild_card, pos, ActionType.PLACE)
            try:
                next_state = state.apply_action(action)
                score = scoring_fn.evaluate(next_state, team)
            except (ValueError, AssertionError):
                score = 0.0

            scores.append(score)
            positions.append((r, c))

    if not scores:
        return heatmap

    # Normalize to [0, 1]
    min_score = min(scores)
    max_score = max(scores)
    score_range = max_score - min_score

    for (r, c), score in zip(positions, scores):
        if score_range > 0:
            heatmap[r, c] = (score - min_score) / score_range
        else:
            heatmap[r, c] = 0.5

    return heatmap
