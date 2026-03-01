"""Analysis modules for Sequence game data."""

from .explainer import explain_weights, generate_report, rank_tips_by_importance
from .heatmaps import (
    mcts_attention,
    placement_frequency,
    sequence_participation,
    win_contribution,
)
from .statistics import (
    average_game_length,
    compute_elo,
    first_player_advantage,
    print_tournament_results,
    win_rate_with_ci,
)

__all__ = [
    "average_game_length",
    "compute_elo",
    "explain_weights",
    "first_player_advantage",
    "generate_report",
    "mcts_attention",
    "placement_frequency",
    "print_tournament_results",
    "rank_tips_by_importance",
    "sequence_participation",
    "win_contribution",
    "win_rate_with_ci",
]
