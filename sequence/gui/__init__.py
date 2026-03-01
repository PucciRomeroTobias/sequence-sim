"""Sequence game GUI built with tkinter."""

from .app import SequenceApp, main
from .board_canvas import BoardCanvas
from .heatmap_view import compute_scoring_heatmap
from .replay_view import ReplayView

__all__ = [
    "BoardCanvas",
    "ReplayView",
    "SequenceApp",
    "compute_scoring_heatmap",
    "main",
]
