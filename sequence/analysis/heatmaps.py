"""Heatmap analysis for Sequence game records."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..core.game import GameRecord


def placement_frequency(
    records: list[GameRecord], team: int | None = None
) -> np.ndarray:
    """Count how often each board position was played.

    Args:
        records: List of game records.
        team: If specified, only count placements by this team. Otherwise all.

    Returns:
        10x10 ndarray of placement counts.
    """
    freq = np.zeros((10, 10), dtype=np.float64)
    for record in records:
        for move in record.moves:
            if team is not None and move.team != team:
                continue
            pos = move.action.get("position")
            if pos is not None and len(pos) == 2:
                freq[pos[0], pos[1]] += 1
    return freq


def win_contribution(records: list[GameRecord]) -> np.ndarray:
    """Compute correlation between position placement and victory.

    For each position, compute: (times played by winner) - (times played by loser),
    normalized by total games with a winner.

    Returns:
        10x10 ndarray of win contribution scores.
    """
    contrib = np.zeros((10, 10), dtype=np.float64)
    games_with_winner = 0

    for record in records:
        if record.winner is None:
            continue
        games_with_winner += 1
        for move in record.moves:
            pos = move.action.get("position")
            if pos is None or len(pos) != 2:
                continue
            if move.team == record.winner:
                contrib[pos[0], pos[1]] += 1
            else:
                contrib[pos[0], pos[1]] -= 1

    if games_with_winner > 0:
        contrib /= games_with_winner
    return contrib


def mcts_attention(records: list[GameRecord]) -> np.ndarray:
    """Compute average MCTS visit counts per board position.

    Uses the mcts_visits dict from MoveRecords that have it.

    Returns:
        10x10 ndarray of average visit counts.
    """
    total_visits = np.zeros((10, 10), dtype=np.float64)
    count = 0

    for record in records:
        for move in record.moves:
            if move.mcts_visits is None:
                continue
            count += 1
            for key, visits in move.mcts_visits.items():
                # Keys are stringified positions like "3,4" or "(3, 4)"
                pos = _parse_position_key(key)
                if pos is not None:
                    total_visits[pos[0], pos[1]] += visits

    if count > 0:
        total_visits /= count
    return total_visits


def sequence_participation(records: list[GameRecord]) -> np.ndarray:
    """Count how often each position was part of a newly completed sequence.

    Detects sequence completions by comparing sequences_before and sequences_after
    in each move record.

    Returns:
        10x10 ndarray of participation counts.
    """
    participation = np.zeros((10, 10), dtype=np.float64)

    for record in records:
        for move in record.moves:
            team = move.team
            before = move.sequences_before.get(team, 0) if isinstance(
                move.sequences_before, dict
            ) else 0
            # Handle both int and str keys from JSON deserialization
            after_val = move.sequences_after
            if isinstance(after_val, dict):
                after = after_val.get(team, after_val.get(str(team), 0))
            else:
                after = 0
            before_val = move.sequences_before
            if isinstance(before_val, dict):
                before = before_val.get(team, before_val.get(str(team), 0))
            else:
                before = 0

            if after > before:
                # A sequence was completed this turn.
                # The played position is part of it.
                pos = move.action.get("position")
                if pos is not None and len(pos) == 2:
                    participation[pos[0], pos[1]] += 1
                # Also credit the other positions in the board snapshot
                # that belong to this team around the played position.
                # For simplicity, just credit the played position.

    return participation


def _parse_position_key(key: str) -> tuple[int, int] | None:
    """Parse a position key string into (row, col).

    Handles formats:
    - "3,4", "(3, 4)", "(3,4)"
    - "Place(QC@Position(row=6, col=0))"
    - "Remove(JH@Position(row=3, col=5))"
    """
    import re

    # Try "Position(row=X, col=Y)" format first
    m = re.search(r'row=(\d+),\s*col=(\d+)', key)
    if m:
        return (int(m.group(1)), int(m.group(2)))

    # Try simple "3,4" or "(3, 4)" format
    cleaned = key.strip().strip("()")
    parts = cleaned.split(",")
    if len(parts) == 2:
        try:
            return (int(parts[0].strip()), int(parts[1].strip()))
        except ValueError:
            return None
    return None
