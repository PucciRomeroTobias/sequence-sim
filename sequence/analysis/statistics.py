"""Statistical analysis utilities for Sequence game records."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.game import GameRecord


def win_rate_with_ci(
    wins: int, total: int, confidence: float = 0.95
) -> tuple[float, float, float]:
    """Wilson score interval for a binomial proportion.

    Returns (rate, lower, upper).
    """
    if total == 0:
        return (0.0, 0.0, 0.0)
    # z-score for confidence level
    z = _z_score(confidence)
    p = wins / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denominator
    lower = max(0.0, centre - margin)
    upper = min(1.0, centre + margin)
    return (p, lower, upper)


def _z_score(confidence: float) -> float:
    """Approximate z-score for common confidence levels."""
    # Use the most common values; fall back to 1.96 for 0.95
    table = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
    return table.get(confidence, 1.96)


def compute_elo(
    results: list[tuple[str, str, float]], k: float = 32, initial: float = 1500
) -> dict[str, float]:
    """Compute Elo ratings from a list of (player_a, player_b, score_a) tuples.

    score_a is 1.0 for a win, 0.0 for a loss, 0.5 for a draw.
    Returns dict mapping player name -> final Elo rating.
    """
    ratings: dict[str, float] = {}
    for player_a, player_b, score_a in results:
        if player_a not in ratings:
            ratings[player_a] = initial
        if player_b not in ratings:
            ratings[player_b] = initial

        ra = ratings[player_a]
        rb = ratings[player_b]
        ea = 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))
        eb = 1.0 - ea

        ratings[player_a] = ra + k * (score_a - ea)
        ratings[player_b] = rb + k * ((1.0 - score_a) - eb)

    return ratings


def first_player_advantage(records: list[GameRecord]) -> float:
    """Return the percentage of games won by team 0 (first player)."""
    if not records:
        return 0.0
    wins = sum(1 for r in records if r.winner == 0)
    return wins / len(records)


def average_game_length(records: list[GameRecord]) -> float:
    """Return the average number of turns across game records."""
    if not records:
        return 0.0
    return sum(r.total_turns for r in records) / len(records)


def print_tournament_results(
    agent_names: list[str], win_matrix: list[list[int]]
) -> str:
    """Format tournament results as a readable table.

    win_matrix[i][j] = number of times agent i beat agent j.
    Returns the formatted string.
    """
    # Column width: at least as wide as longest name + padding
    col_w = max(len(n) for n in agent_names) + 2
    num_w = 6

    lines: list[str] = []
    # Header
    header = " " * col_w + "".join(n.center(num_w) for n in agent_names)
    lines.append(header)
    lines.append("-" * len(header))

    for i, name in enumerate(agent_names):
        row_parts = [name.ljust(col_w)]
        for j in range(len(agent_names)):
            if i == j:
                row_parts.append("  -   ")
            else:
                row_parts.append(str(win_matrix[i][j]).center(num_w))
        lines.append("".join(row_parts))

    result = "\n".join(lines)
    return result
