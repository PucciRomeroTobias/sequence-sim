"""Board canvas widget for rendering the Sequence game board."""

from __future__ import annotations

import tkinter as tk
from typing import Any

from ..core.board import LAYOUT
from ..core.types import CORNER, CORNERS, EMPTY, Position

# Board dimensions
CELL_SIZE = 60
BOARD_PX = CELL_SIZE * 10

# Chip colors per team
TEAM_COLORS: dict[int, str] = {
    0: "#E74C3C",  # Red
    1: "#3498DB",  # Blue
    2: "#2ECC71",  # Green
}

CORNER_COLOR = "#F1C40F"  # Gold

# Card label for display
_LAYOUT_STRINGS: list[list[str]] = []
for row in LAYOUT:
    _row: list[str] = []
    for card in row:
        _row.append(str(card) if card is not None else "FREE")
    _LAYOUT_STRINGS.append(_row)


class BoardCanvas(tk.Canvas):
    """Canvas that renders the Sequence game board."""

    def __init__(self, parent: tk.Widget, **kwargs: Any) -> None:
        kwargs.setdefault("width", BOARD_PX)
        kwargs.setdefault("height", BOARD_PX)
        kwargs.setdefault("bg", "#F5F5DC")
        super().__init__(parent, **kwargs)
        self._heatmap: list[list[float]] | None = None
        self._draw_empty_board()

    def _draw_empty_board(self) -> None:
        """Draw the initial board grid and card labels."""
        self.delete("all")
        for r in range(10):
            for c in range(10):
                x0 = c * CELL_SIZE
                y0 = r * CELL_SIZE
                x1 = x0 + CELL_SIZE
                y1 = y0 + CELL_SIZE

                # Corner cells
                pos = Position(r, c)
                if pos in CORNERS:
                    self.create_rectangle(
                        x0, y0, x1, y1, fill=CORNER_COLOR, outline="#888", width=1
                    )
                    self.create_text(
                        x0 + CELL_SIZE // 2,
                        y0 + CELL_SIZE // 2,
                        text="FREE",
                        font=("Helvetica", 8, "bold"),
                        fill="#333",
                    )
                else:
                    self.create_rectangle(
                        x0, y0, x1, y1, fill="#F5F5DC", outline="#888", width=1
                    )
                    self.create_text(
                        x0 + CELL_SIZE // 2,
                        y0 + CELL_SIZE // 2,
                        text=_LAYOUT_STRINGS[r][c],
                        font=("Helvetica", 8),
                        fill="#555",
                    )

    def update_board(
        self,
        board: Any,
        last_action: dict[str, Any] | None = None,
        highlights: list[Position] | None = None,
    ) -> None:
        """Redraw the board with current chip state.

        Args:
            board: Board object with .chips ndarray
            last_action: Serialized action dict with 'position' key
            highlights: Positions to highlight (e.g. legal moves)
        """
        self.delete("all")
        chips = board.chips

        for r in range(10):
            for c in range(10):
                x0 = c * CELL_SIZE
                y0 = r * CELL_SIZE
                x1 = x0 + CELL_SIZE
                y1 = y0 + CELL_SIZE
                cx = x0 + CELL_SIZE // 2
                cy = y0 + CELL_SIZE // 2
                pos = Position(r, c)

                # Determine cell background
                fill = "#F5F5DC"
                outline = "#888"
                outline_width = 1

                if pos in CORNERS:
                    fill = CORNER_COLOR

                # Heatmap overlay
                if self._heatmap is not None:
                    val = self._heatmap[r][c]
                    fill = _heatmap_color(val)

                # Highlight legal positions
                if highlights and pos in highlights:
                    fill = "#ABEBC6"  # Light green

                # Highlight last played position
                if last_action and last_action.get("position"):
                    lp = last_action["position"]
                    if lp[0] == r and lp[1] == c:
                        outline = "#F4D03F"  # Yellow border
                        outline_width = 3

                self.create_rectangle(
                    x0, y0, x1, y1, fill=fill, outline=outline, width=outline_width
                )

                # Card label
                label = _LAYOUT_STRINGS[r][c]
                self.create_text(
                    cx, cy - 10, text=label, font=("Helvetica", 7), fill="#555"
                )

                # Draw chip
                chip_val = int(chips[r, c])
                if chip_val == CORNER:
                    self.create_oval(
                        cx - 12, cy, cx + 12, cy + 20, fill=CORNER_COLOR, outline="#B7950B", width=2
                    )
                elif chip_val != EMPTY:
                    color = TEAM_COLORS.get(chip_val, "#999")
                    self.create_oval(
                        cx - 12, cy, cx + 12, cy + 20, fill=color, outline="#333", width=2
                    )

    def set_heatmap(self, values_10x10: list[list[float]] | None) -> None:
        """Set a 10x10 heatmap for colored overlay. None clears it."""
        self._heatmap = values_10x10

    def clear_heatmap(self) -> None:
        self._heatmap = None


def _heatmap_color(value: float) -> str:
    """Map a 0..1 value to a red-yellow-green color string."""
    value = max(0.0, min(1.0, value))
    if value < 0.5:
        # Red to yellow
        t = value * 2
        r = 255
        g = int(255 * t)
        b = 0
    else:
        # Yellow to green
        t = (value - 0.5) * 2
        r = int(255 * (1 - t))
        g = 255
        b = 0
    return f"#{r:02x}{g:02x}{b:02x}"
