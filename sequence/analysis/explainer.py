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
    # Card-tracking features
    "guaranteed_positions": "If you hold both copies of a card, that position is guaranteed — build lines through it",
    "viable_four_in_a_row": "A 4-in-a-row only matters if the completing card still exists",
    "viable_three_in_a_row": "Focus on lines where the cards you need are still in play",
    "opp_blockable_four": "Block opponent 4-in-a-row immediately when you can",
    "opp_unblockable_four": "When you can't block, focus on winning faster instead",
    "fork_count": "Fork positions advance 2+ sequences at once — forces impossible choices",
    "dead_positions_for_us": "Avoid investing in lines through permanently dead positions",
    "dead_lines": "Don't waste moves on lines that can never be completed",
    "opp_dead_lines": "You don't need to block lines the opponent can never finish",
    "card_monopoly_count": "Card monopolies = guaranteed positions. Use them strategically",
    "sequence_proximity": "Push your most advanced viable line to completion",
    # Advanced strategy features
    "position_line_score": "Prefer positions that participate in many possible lines (center > edges)",
    "anchor_overlap_count": "Chips in completed sequences can anchor a second sequence through shared lines",
    "chip_clustering": "Concentrate chips in one board region instead of spreading them out",
    "early_jack_usage_penalty": "Penalty for using jacks early — save them for critical moments",
    "jack_save_value": "Save jacks for critical moments — their value is highest in early game",
}

# Human tips for card-counting strategy
_CARD_COUNTING_TIPS: list[str] = [
    "Track which cards have been played — each card has only 2 copies in the deck",
    "When both copies of a card are used, positions needing it are dead (unless you have a two-eyed jack)",
    "If you hold both remaining copies of a card, you have a monopoly on those board positions",
    "Count two-eyed jacks carefully — they're the most flexible cards in the game",
    "Watch what your opponent discards as dead cards — this reveals their hand weakness",
    "Concentrate chips in one area of the board instead of spreading them out — clustering creates more overlapping lines",
]


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

    # Card counting tips
    weight_dict = {f.name: getattr(weights, f.name) for f in fields(weights)}
    has_tracking_weights = any(
        abs(weight_dict.get(name, 0)) > 0
        for name in (
            "guaranteed_positions", "viable_four_in_a_row",
            "card_monopoly_count", "dead_lines", "fork_count",
        )
    )
    if has_tracking_weights:
        lines.append("--- Card Counting Strategy ---")
        for i, tip in enumerate(_CARD_COUNTING_TIPS, 1):
            lines.append(f"  {i}. {tip}")
        lines.append("")

    # Tactical priorities
    lines.append("--- Tactical Priorities ---")
    tactics: list[tuple[str, float]] = []
    if abs(weight_dict.get("fork_count", 0)) > 0:
        tactics.append(("Create fork positions that advance multiple lines at once", weight_dict["fork_count"]))
    if abs(weight_dict.get("viable_four_in_a_row", 0)) > 0:
        tactics.append(("Focus on 4-in-a-row lines where the completing card exists", weight_dict["viable_four_in_a_row"]))
    if abs(weight_dict.get("opp_blockable_four", 0)) > 0:
        tactics.append(("Block opponent 4-in-a-row when you hold a blocking card", weight_dict["opp_blockable_four"]))
    if abs(weight_dict.get("guaranteed_positions", 0)) > 0:
        tactics.append(("Build lines through positions you've monopolized", weight_dict["guaranteed_positions"]))
    if abs(weight_dict.get("dead_lines", 0)) > 0:
        tactics.append(("Abandon lines that can no longer be completed", weight_dict["dead_lines"]))
    if abs(weight_dict.get("sequence_proximity", 0)) > 0:
        tactics.append(("Push your most advanced viable line to completion", weight_dict["sequence_proximity"]))

    tactics.sort(key=lambda x: abs(x[1]), reverse=True)
    for i, (tip, _) in enumerate(tactics, 1):
        lines.append(f"  {i}. {tip}")
    if not tactics:
        lines.append("  (No card-tracking tactics active)")
    lines.append("")

    # Hand management advice
    lines.append("--- Hand Management ---")
    hand_tips: list[str] = []
    if weight_dict.get("dead_cards_in_hand", 0) < -1:
        hand_tips.append("Discard dead cards quickly to cycle your hand")
    if weight_dict.get("two_eyed_jacks_in_hand", 0) > 1:
        hand_tips.append("Save two-eyed jacks for critical moments (completing sequences, filling dead positions)")
    if weight_dict.get("one_eyed_jacks_in_hand", 0) > 1:
        hand_tips.append("Save one-eyed jacks for disrupting opponent 4-in-a-row threats")
    if weight_dict.get("card_monopoly_count", 0) > 1:
        hand_tips.append("Hold both copies of a card when possible for guaranteed placement")
    if weight_dict.get("hand_pairs", 0) > 0:
        hand_tips.append("Card pairs give you flexibility — you control both board positions for that card")
    for i, tip in enumerate(hand_tips, 1):
        lines.append(f"  {i}. {tip}")
    if not hand_tips:
        lines.append("  (No specific hand management advice)")
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
