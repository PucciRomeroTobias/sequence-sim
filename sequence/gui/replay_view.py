"""Replay view for navigating recorded games."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Any

from ..core.board import Board
from ..core.game import GameRecord, MoveRecord
from ..core.types import Position, TeamId
from .board_canvas import BoardCanvas


class ReplayView(tk.Frame):
    """Frame for replaying a recorded game with navigation controls."""

    def __init__(self, parent: tk.Widget, **kwargs: Any) -> None:
        super().__init__(parent, **kwargs)
        self._record: GameRecord | None = None
        self._turn_index: int = 0
        self._playing: bool = False
        self._play_after_id: str | None = None
        self._speed: float = 1.0

        self._build_ui()

    def _build_ui(self) -> None:
        # Board canvas
        self._canvas = BoardCanvas(self)
        self._canvas.pack(side=tk.LEFT, padx=5, pady=5)

        # Right panel
        right = tk.Frame(self)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Info panel
        info_frame = tk.LabelFrame(right, text="Game Info", padx=5, pady=5)
        info_frame.pack(fill=tk.X, pady=(0, 5))
        self._info_label = tk.Label(
            info_frame, text="No game loaded", justify=tk.LEFT, anchor="w", font=("Courier", 10)
        )
        self._info_label.pack(fill=tk.X)

        # Slider
        slider_frame = tk.Frame(right)
        slider_frame.pack(fill=tk.X, pady=5)
        self._slider = tk.Scale(
            slider_frame,
            from_=0,
            to=0,
            orient=tk.HORIZONTAL,
            command=self._on_slider,
            label="Turn",
        )
        self._slider.pack(fill=tk.X)

        # Navigation buttons
        btn_frame = tk.Frame(right)
        btn_frame.pack(fill=tk.X, pady=5)

        self._btn_first = tk.Button(btn_frame, text="|<", width=4, command=self._go_first)
        self._btn_first.pack(side=tk.LEFT, padx=2)

        self._btn_prev = tk.Button(btn_frame, text="<", width=4, command=self._go_prev)
        self._btn_prev.pack(side=tk.LEFT, padx=2)

        self._btn_play = tk.Button(btn_frame, text="Play", width=6, command=self._toggle_play)
        self._btn_play.pack(side=tk.LEFT, padx=2)

        self._btn_next = tk.Button(btn_frame, text=">", width=4, command=self._go_next)
        self._btn_next.pack(side=tk.LEFT, padx=2)

        self._btn_last = tk.Button(btn_frame, text=">|", width=4, command=self._go_last)
        self._btn_last.pack(side=tk.LEFT, padx=2)

        # Speed control
        speed_frame = tk.Frame(right)
        speed_frame.pack(fill=tk.X, pady=5)
        tk.Label(speed_frame, text="Speed:").pack(side=tk.LEFT, padx=2)
        self._speed_var = tk.StringVar(value="1x")
        speed_menu = ttk.Combobox(
            speed_frame,
            textvariable=self._speed_var,
            values=["0.5x", "1x", "2x", "5x"],
            width=5,
            state="readonly",
        )
        speed_menu.pack(side=tk.LEFT, padx=2)
        speed_menu.bind("<<ComboboxSelected>>", self._on_speed_change)

        # Move detail panel
        detail_frame = tk.LabelFrame(right, text="Move Detail", padx=5, pady=5)
        detail_frame.pack(fill=tk.BOTH, expand=True, pady=(5, 0))
        self._detail_label = tk.Label(
            detail_frame, text="", justify=tk.LEFT, anchor="nw", font=("Courier", 9)
        )
        self._detail_label.pack(fill=tk.BOTH, expand=True)

    def load_record(self, record: GameRecord) -> None:
        """Load a game record for replay."""
        self._stop_play()
        self._record = record
        self._turn_index = 0
        max_turn = max(0, len(record.moves) - 1)
        self._slider.configure(to=max_turn)
        self._slider.set(0)
        self._render_turn()

    def _on_slider(self, value: str) -> None:
        idx = int(float(value))
        if idx != self._turn_index:
            self._turn_index = idx
            self._render_turn()

    def _go_first(self) -> None:
        self._stop_play()
        self._turn_index = 0
        self._slider.set(0)
        self._render_turn()

    def _go_prev(self) -> None:
        self._stop_play()
        if self._turn_index > 0:
            self._turn_index -= 1
            self._slider.set(self._turn_index)
            self._render_turn()

    def _go_next(self) -> None:
        if self._record and self._turn_index < len(self._record.moves) - 1:
            self._turn_index += 1
            self._slider.set(self._turn_index)
            self._render_turn()
        else:
            self._stop_play()

    def _go_last(self) -> None:
        self._stop_play()
        if self._record:
            self._turn_index = len(self._record.moves) - 1
            self._slider.set(self._turn_index)
            self._render_turn()

    def _toggle_play(self) -> None:
        if self._playing:
            self._stop_play()
        else:
            self._start_play()

    def _start_play(self) -> None:
        self._playing = True
        self._btn_play.configure(text="Pause")
        self._play_step()

    def _stop_play(self) -> None:
        self._playing = False
        self._btn_play.configure(text="Play")
        if self._play_after_id is not None:
            self.after_cancel(self._play_after_id)
            self._play_after_id = None

    def _play_step(self) -> None:
        if not self._playing or self._record is None:
            self._stop_play()
            return
        if self._turn_index < len(self._record.moves) - 1:
            self._turn_index += 1
            self._slider.set(self._turn_index)
            self._render_turn()
            delay = int(500 / self._speed)
            self._play_after_id = self.after(delay, self._play_step)
        else:
            self._stop_play()

    def _on_speed_change(self, _event: Any) -> None:
        text = self._speed_var.get()
        self._speed = float(text.replace("x", ""))

    def _render_turn(self) -> None:
        if self._record is None:
            return

        record = self._record
        moves = record.moves

        if not moves:
            self._info_label.configure(text="No moves in this game")
            return

        move = moves[self._turn_index]

        # Reconstruct board from snapshot
        board = Board.from_list(move.board_snapshot)
        last_action = move.action

        self._canvas.update_board(board, last_action=last_action)

        # Game info
        winner_str = f"Team {record.winner}" if record.winner is not None else "Draw"
        agents_str = ", ".join(record.agent_names)
        info = (
            f"Game: {record.game_id}\n"
            f"Agents: {agents_str}\n"
            f"Winner: {winner_str}\n"
            f"Turns: {record.total_turns}"
        )
        self._info_label.configure(text=info)

        # Move detail
        action = move.action
        action_type = action.get("action_type", "?")
        card = action.get("card", "?")
        pos = action.get("position")
        pos_str = f"({pos[0]},{pos[1]})" if pos else "N/A"
        team = move.team

        hand_str = ", ".join(move.hand_before[:7])

        seq_before = move.sequences_before
        seq_after = move.sequences_after
        seq_str = "  ".join(
            f"T{t}:{seq_before.get(str(t), seq_before.get(t, 0))}->{seq_after.get(str(t), seq_after.get(t, 0))}"
            for t in range(record.config.get("num_teams", 2))
        )

        detail = (
            f"Turn {move.turn} | Team {team}\n"
            f"Action: {action_type}\n"
            f"Card: {card}  Pos: {pos_str}\n"
            f"Hand: {hand_str}\n"
            f"Sequences: {seq_str}\n"
            f"Legal actions: {move.legal_actions_count}"
        )
        self._detail_label.configure(text=detail)
