"""Strategy explainer for Sequence scoring weights."""

from __future__ import annotations

from dataclasses import fields
from typing import TYPE_CHECKING

from ..scoring.scoring_function import FEATURE_NAMES, ScoringWeights

if TYPE_CHECKING:
    from ..core.game import GameRecord

# Human-readable descriptions for each feature weight
_FEATURE_DESCRIPTIONS: dict[str, str] = {
    "completed_sequences": "Reward for completing a 5-in-a-row sequence",
    "four_in_a_row": "Reward for having 4 chips in a line (one away from sequence)",
    "three_in_a_row": "Reward for having 3 chips in a line",
    "two_in_a_row": "Reward for having 2 chips in a line",
    "opp_completed_sequences": "Penalty when opponent completes a sequence",
    "opp_four_in_a_row": "Penalty for opponent having 4-in-a-row",
    "opp_three_in_a_row": "Penalty for opponent having 3-in-a-row",
    "chips_on_board": "Value of own chip presence on the board",
    "opp_chips_on_board": "Penalty for opponent chip presence",
    "center_control": "Value of controlling center positions",
    "corner_adjacency": "Value of placing chips adjacent to corners",
    "hand_pairs": "Value of holding card pairs in hand",
    "two_eyed_jacks_in_hand": "Value of two-eyed jacks (wild placement)",
    "one_eyed_jacks_in_hand": "Value of one-eyed jacks (removal)",
    "dead_cards_in_hand": "Penalty for unplayable cards in hand",
    "shared_line_potential": "Value of positions contributing to multiple lines",
    "blocked_lines": "Penalty for lines blocked by opponent",
}


def explain_weights(weights: ScoringWeights) -> list[str]:
    """Generate human-readable strategy tips from a weight vector.

    Returns a list of advice strings sorted by weight magnitude (most impactful first).
    """
    ranked = rank_tips_by_importance(weights)
    tips: list[str] = []
    for name, value in ranked:
        desc = _FEATURE_DESCRIPTIONS.get(name, name.replace("_", " "))
        if value > 0:
            tips.append(f"{desc} (weight: +{value:.0f}, priority)")
        elif value < 0:
            tips.append(f"{desc} (weight: {value:.0f}, avoid/block)")
        else:
            tips.append(f"{desc} (weight: 0, ignored)")

    # Add comparative insights
    weight_dict = {f.name: getattr(weights, f.name) for f in fields(weights)}
    comparisons = _generate_comparisons(weight_dict)
    tips.extend(comparisons)

    return tips


def rank_tips_by_importance(weights: ScoringWeights) -> list[tuple[str, float]]:
    """Return (feature_name, weight_value) pairs sorted by absolute magnitude."""
    pairs: list[tuple[str, float]] = []
    for f in fields(weights):
        pairs.append((f.name, getattr(weights, f.name)))
    pairs.sort(key=lambda x: abs(x[1]), reverse=True)
    return pairs


def _generate_comparisons(weight_dict: dict[str, float]) -> list[str]:
    """Generate comparative insights between related weights."""
    comparisons: list[str] = []

    # Compare offensive vs defensive posture
    offense = abs(weight_dict.get("four_in_a_row", 0))
    defense = abs(weight_dict.get("opp_four_in_a_row", 0))
    if offense > 0 and defense > 0:
        if defense > offense * 1.5:
            comparisons.append(
                "Defensive posture: blocking opponent 4-in-a-row is prioritized "
                f"over building own ({defense:.0f} vs {offense:.0f})"
            )
        elif offense > defense * 1.5:
            comparisons.append(
                "Aggressive posture: building 4-in-a-row is prioritized "
                f"over blocking opponent ({offense:.0f} vs {defense:.0f})"
            )

    # Compare center vs corner
    center = abs(weight_dict.get("center_control", 0))
    corner = abs(weight_dict.get("corner_adjacency", 0))
    if center > 0 and corner > 0:
        ratio = center / corner if corner != 0 else float("inf")
        if ratio > 1.5:
            comparisons.append(
                f"Center control is {ratio:.1f}x more important than corner adjacency"
            )
        elif ratio < 0.67:
            comparisons.append(
                f"Corner adjacency is {1/ratio:.1f}x more important than center control"
            )

    return comparisons


def generate_report(
    weights: ScoringWeights, records: list[GameRecord] | None = None
) -> str:
    """Generate a full strategy report from weights and optional game records.

    Returns a multi-line report string.
    """
    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("SEQUENCE STRATEGY REPORT")
    lines.append("=" * 60)
    lines.append("")

    # Weight analysis
    lines.append("--- Weight Analysis ---")
    tips = explain_weights(weights)
    for i, tip in enumerate(tips, 1):
        lines.append(f"  {i}. {tip}")
    lines.append("")

    # Top priorities
    ranked = rank_tips_by_importance(weights)
    lines.append("--- Top 5 Priorities ---")
    for i, (name, value) in enumerate(ranked[:5], 1):
        desc = _FEATURE_DESCRIPTIONS.get(name, name)
        lines.append(f"  {i}. {desc}: {value:+.1f}")
    lines.append("")

    # Game statistics if records provided
    if records:
        from .statistics import average_game_length, first_player_advantage

        lines.append("--- Game Statistics ---")
        lines.append(f"  Games analyzed: {len(records)}")
        lines.append(f"  Average game length: {average_game_length(records):.1f} turns")
        lines.append(
            f"  First player advantage: {first_player_advantage(records):.1%}"
        )
        wins = sum(1 for r in records if r.winner is not None)
        draws = len(records) - wins
        lines.append(f"  Decisive games: {wins}/{len(records)} ({wins/len(records):.1%})")
        if draws > 0:
            lines.append(f"  Draws: {draws}")
        lines.append("")

    lines.append("=" * 60)
    return "\n".join(lines)
