"""Game phase analysis: early/mid/late game insights."""

from __future__ import annotations

import numpy as np

from ..core.board import ALL_LINES, LAYOUT
from ..core.game import GameRecord, MoveRecord
from ..core.types import CORNER, CORNERS, Position, TeamId


def classify_phase(turn: int, total_turns: int) -> str:
    """Classify a turn into early/mid/late game phase."""
    ratio = turn / max(total_turns, 1)
    if ratio < 0.33:
        return "early"
    elif ratio < 0.66:
        return "mid"
    else:
        return "late"


def placement_by_phase(records: list[GameRecord]) -> dict[str, np.ndarray]:
    """Compute placement frequency heatmaps per game phase.

    Returns dict with keys 'early', 'mid', 'late', each a 10x10 array.
    """
    maps = {phase: np.zeros((10, 10)) for phase in ("early", "mid", "late")}

    for record in records:
        total = record.total_turns
        for move in record.moves:
            pos_data = move.action.get("position")
            if pos_data and move.action.get("action_type") == "place":
                r, c = pos_data
                phase = classify_phase(move.turn, total)
                maps[phase][r, c] += 1

    # Normalize each to [0, 1]
    for phase in maps:
        mx = maps[phase].max()
        if mx > 0:
            maps[phase] /= mx

    return maps


def jack_timing(records: list[GameRecord]) -> dict[str, dict[str, float]]:
    """Analyze when Jacks are played relative to game phase.

    Returns dict mapping Jack type to phase distribution.
    """
    counts: dict[str, dict[str, int]] = {
        "two_eyed": {"early": 0, "mid": 0, "late": 0},
        "one_eyed": {"early": 0, "mid": 0, "late": 0},
    }

    for record in records:
        total = record.total_turns
        for move in record.moves:
            card_str = move.action.get("card", "")
            if not card_str:
                continue
            # Jacks: JS, JH = one-eyed; JD, JC = two-eyed
            if card_str.startswith("J"):
                suit = card_str[1] if len(card_str) == 2 else ""
                phase = classify_phase(move.turn, total)
                if suit in ("D", "C"):
                    counts["two_eyed"][phase] += 1
                elif suit in ("H", "S"):
                    counts["one_eyed"][phase] += 1

    # Convert to percentages
    result: dict[str, dict[str, float]] = {}
    for jtype, phases in counts.items():
        total = sum(phases.values())
        if total > 0:
            result[jtype] = {p: c / total for p, c in phases.items()}
        else:
            result[jtype] = {p: 0.0 for p in phases}

    return result


def win_rate_by_phase_control(
    records: list[GameRecord],
) -> dict[str, float]:
    """Analyze how board control at each phase correlates with winning.

    Returns dict of 'early_lead_wins', 'mid_lead_wins', 'late_lead_wins'.
    """
    phase_lead_wins = {"early": 0, "mid": 0, "late": 0}
    phase_lead_total = {"early": 0, "mid": 0, "late": 0}

    for record in records:
        if record.winner is None:
            continue
        total = record.total_turns
        for move in record.moves:
            phase = classify_phase(move.turn, total)
            # Count chips per team from board snapshot
            snapshot = move.board_snapshot
            team0_chips = sum(1 for r in snapshot for v in r if v == 0)
            team1_chips = sum(1 for r in snapshot for v in r if v == 1)

            if team0_chips != team1_chips:
                leader = 0 if team0_chips > team1_chips else 1
                phase_lead_total[phase] += 1
                if leader == record.winner:
                    phase_lead_wins[phase] += 1

    return {
        f"{phase}_lead_win_rate": (
            phase_lead_wins[phase] / phase_lead_total[phase]
            if phase_lead_total[phase] > 0
            else 0.5
        )
        for phase in ("early", "mid", "late")
    }


def center_vs_edge_analysis(records: list[GameRecord]) -> dict[str, float]:
    """Analyze whether winners place more chips in center vs edges."""
    center_rows = range(2, 8)
    center_cols = range(2, 8)

    winner_center_ratio = []
    loser_center_ratio = []

    for record in records:
        if record.winner is None:
            continue
        # Analyze final board state (last move's board)
        if not record.moves:
            continue
        final_board = record.moves[-1].board_snapshot

        for team in range(2):
            center = sum(
                1
                for r in center_rows
                for c in center_cols
                if final_board[r][c] == team
            )
            total = sum(1 for r in range(10) for c in range(10) if final_board[r][c] == team)
            if total > 0:
                ratio = center / total
                if team == record.winner:
                    winner_center_ratio.append(ratio)
                else:
                    loser_center_ratio.append(ratio)

    return {
        "winner_center_ratio": (
            sum(winner_center_ratio) / len(winner_center_ratio)
            if winner_center_ratio
            else 0.0
        ),
        "loser_center_ratio": (
            sum(loser_center_ratio) / len(loser_center_ratio)
            if loser_center_ratio
            else 0.0
        ),
    }


def opening_moves_analysis(
    records: list[GameRecord], depth: int = 3
) -> list[tuple[str, int, float]]:
    """Analyze the most common opening moves and their win rates.

    Returns list of (move_description, count, win_rate) sorted by count.
    """
    from collections import Counter

    opening_wins: dict[str, int] = Counter()
    opening_total: dict[str, int] = Counter()

    for record in records:
        if record.winner is None or len(record.moves) < depth:
            continue
        # First `depth` moves as key
        key_parts = []
        for move in record.moves[:depth]:
            pos = move.action.get("position")
            if pos:
                key_parts.append(f"({pos[0]},{pos[1]})")
            else:
                key_parts.append("discard")

        key = " -> ".join(key_parts)
        opening_total[key] += 1
        if record.winner == 0:
            opening_wins[key] += 1

    result = []
    for key, total in opening_total.most_common(20):
        wins = opening_wins.get(key, 0)
        result.append((key, total, wins / total if total > 0 else 0.5))

    return result


def generate_phase_report(records: list[GameRecord]) -> str:
    """Generate a comprehensive game phase analysis report."""
    lines = []
    lines.append("=" * 60)
    lines.append("  GAME PHASE ANALYSIS")
    lines.append("=" * 60)

    if not records:
        lines.append("  No game records to analyze.")
        return "\n".join(lines)

    # Jack timing
    jt = jack_timing(records)
    lines.append("\n  JACK TIMING:")
    for jtype, phases in jt.items():
        name = "Two-eyed (wild)" if jtype == "two_eyed" else "One-eyed (removal)"
        lines.append(f"    {name}:")
        for phase, pct in phases.items():
            bar = "#" * int(pct * 30)
            lines.append(f"      {phase:>5}: {pct:5.1%} {bar}")

    # Phase control correlation
    pcw = win_rate_by_phase_control(records)
    lines.append("\n  CHIP LEAD -> WIN CORRELATION:")
    for key, rate in pcw.items():
        phase = key.replace("_lead_win_rate", "")
        lines.append(f"    {phase:>5} game lead -> {rate:.1%} win rate")

    # Center vs edge
    ce = center_vs_edge_analysis(records)
    lines.append("\n  CENTER CONTROL:")
    lines.append(
        f"    Winners: {ce['winner_center_ratio']:.1%} of chips in center"
    )
    lines.append(
        f"    Losers:  {ce['loser_center_ratio']:.1%} of chips in center"
    )
    diff = ce["winner_center_ratio"] - ce["loser_center_ratio"]
    if diff > 0.02:
        lines.append(f"    -> Winners have {diff:.1%} more center focus")
    elif diff < -0.02:
        lines.append(f"    -> Losers actually have more center focus ({-diff:.1%})")
    else:
        lines.append("    -> No significant difference")

    # Opening moves
    openings = opening_moves_analysis(records, depth=2)
    if openings:
        lines.append("\n  TOP OPENING SEQUENCES (first 2 moves):")
        for move_desc, count, win_rate in openings[:10]:
            lines.append(f"    {move_desc:>30}  ({count:>3}x, P1 wins {win_rate:.0%})")

    # Game length stats
    lengths = [r.total_turns for r in records]
    lines.append(f"\n  GAME LENGTH:")
    lines.append(f"    Average: {sum(lengths) / len(lengths):.1f} turns")
    lines.append(f"    Min: {min(lengths)}, Max: {max(lengths)}")

    # First player advantage
    p1_wins = sum(1 for r in records if r.winner == 0)
    total_decisive = sum(1 for r in records if r.winner is not None)
    if total_decisive > 0:
        lines.append(f"\n  FIRST PLAYER ADVANTAGE:")
        lines.append(f"    Team 0 wins: {p1_wins}/{total_decisive} ({p1_wins / total_decisive:.1%})")

    return "\n".join(lines)
