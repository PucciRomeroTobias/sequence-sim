"""Main application window for the Sequence game GUI."""

from __future__ import annotations

import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Any

from ..agents import (
    DefensiveAgent,
    GreedyAgent,
    OffensiveAgent,
    RandomAgent,
    ScorerAgent,
)
from ..core.board import Board
from ..core.game import Game, GameConfig, GameRecord
from ..core.types import Position, TeamId
from ..scoring.scoring_function import (
    BALANCED_WEIGHTS,
    DEFENSIVE_WEIGHTS,
    OFFENSIVE_WEIGHTS,
    ScoringFunction,
)
from ..simulation.dataset import DatasetReader
from .board_canvas import BoardCanvas
from .heatmap_view import compute_scoring_heatmap
from .replay_view import ReplayView

# Available agents for selection
AGENT_CHOICES: dict[str, Any] = {
    "Random": lambda: RandomAgent(),
    "Greedy": lambda: GreedyAgent(),
    "Scorer (Balanced)": lambda: ScorerAgent(ScoringFunction(BALANCED_WEIGHTS)),
    "Defensive": lambda: DefensiveAgent(ScoringFunction(DEFENSIVE_WEIGHTS)),
    "Offensive": lambda: OffensiveAgent(ScoringFunction(OFFENSIVE_WEIGHTS)),
}


class SequenceApp(tk.Tk):
    """Main Sequence game application with Live, Replay, and Analysis modes."""

    def __init__(self) -> None:
        super().__init__()
        self.title("Sequence Game Simulator")
        self.geometry("1050x680")
        self.resizable(True, True)

        self._current_record: GameRecord | None = None
        self._game_thread: threading.Thread | None = None
        self._live_playing: bool = False

        self._build_menu()
        self._build_notebook()

    def _build_menu(self) -> None:
        menubar = tk.Menu(self)
        self.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Load JSONL...", command=self._load_jsonl)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.destroy)
        menubar.add_cascade(label="File", menu=file_menu)

    def _build_notebook(self) -> None:
        self._notebook = ttk.Notebook(self)
        self._notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Tab 1: Live game
        self._live_frame = tk.Frame(self._notebook)
        self._notebook.add(self._live_frame, text="Live Game")
        self._build_live_tab()

        # Tab 2: Replay
        self._replay_view = ReplayView(self._notebook)
        self._notebook.add(self._replay_view, text="Replay")

        # Tab 3: Analysis
        self._analysis_frame = tk.Frame(self._notebook)
        self._notebook.add(self._analysis_frame, text="Analysis")
        self._build_analysis_tab()

    # ---- Live Game Tab ----

    def _build_live_tab(self) -> None:
        frame = self._live_frame

        # Board canvas
        self._live_canvas = BoardCanvas(frame)
        self._live_canvas.pack(side=tk.LEFT, padx=5, pady=5)

        # Sidebar
        sidebar = tk.Frame(frame)
        sidebar.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Agent selection
        agent_frame = tk.LabelFrame(sidebar, text="Agent Setup", padx=5, pady=5)
        agent_frame.pack(fill=tk.X, pady=(0, 5))

        agent_names = list(AGENT_CHOICES.keys())

        tk.Label(agent_frame, text="Team 0 (Red):").grid(row=0, column=0, sticky="w")
        self._agent0_var = tk.StringVar(value=agent_names[0])
        ttk.Combobox(
            agent_frame, textvariable=self._agent0_var, values=agent_names, width=18, state="readonly"
        ).grid(row=0, column=1, padx=5, pady=2)

        tk.Label(agent_frame, text="Team 1 (Blue):").grid(row=1, column=0, sticky="w")
        self._agent1_var = tk.StringVar(value=agent_names[1] if len(agent_names) > 1 else agent_names[0])
        ttk.Combobox(
            agent_frame, textvariable=self._agent1_var, values=agent_names, width=18, state="readonly"
        ).grid(row=1, column=1, padx=5, pady=2)

        # Game config
        config_frame = tk.LabelFrame(sidebar, text="Config", padx=5, pady=5)
        config_frame.pack(fill=tk.X, pady=(0, 5))

        tk.Label(config_frame, text="Seed:").grid(row=0, column=0, sticky="w")
        self._seed_var = tk.StringVar(value="42")
        tk.Entry(config_frame, textvariable=self._seed_var, width=10).grid(row=0, column=1, padx=5)

        # Start button
        self._start_btn = tk.Button(sidebar, text="Start Game", command=self._start_live_game)
        self._start_btn.pack(fill=tk.X, pady=5)

        # Game info
        info_frame = tk.LabelFrame(sidebar, text="Game Status", padx=5, pady=5)
        info_frame.pack(fill=tk.BOTH, expand=True)
        self._live_info = tk.Label(
            info_frame, text="Ready to start", justify=tk.LEFT, anchor="nw", font=("Courier", 10)
        )
        self._live_info.pack(fill=tk.BOTH, expand=True)

    def _start_live_game(self) -> None:
        if self._live_playing:
            return

        agent0_name = self._agent0_var.get()
        agent1_name = self._agent1_var.get()

        factory0 = AGENT_CHOICES.get(agent0_name)
        factory1 = AGENT_CHOICES.get(agent1_name)

        if factory0 is None or factory1 is None:
            messagebox.showerror("Error", "Invalid agent selection")
            return

        seed_str = self._seed_var.get().strip()
        seed = int(seed_str) if seed_str.isdigit() else None

        config = GameConfig(num_teams=2, seed=seed)
        game = Game([factory0, factory1], config=config)

        self._start_btn.configure(state=tk.DISABLED)
        self._live_info.configure(text="Running game...")
        self._live_playing = True

        def run_game() -> None:
            try:
                record = game.play()
                self.after(0, lambda: self._on_live_game_done(record))
            except Exception as e:
                self.after(0, lambda: self._on_live_game_error(str(e)))

        self._game_thread = threading.Thread(target=run_game, daemon=True)
        self._game_thread.start()

    def _on_live_game_done(self, record: GameRecord) -> None:
        self._live_playing = False
        self._start_btn.configure(state=tk.NORMAL)
        self._current_record = record

        # Show final board state
        if record.moves:
            last_move = record.moves[-1]
            # Build board from the last move's snapshot, then apply the last action
            board = Board.from_list(last_move.board_snapshot)
            # The snapshot is the state BEFORE the move, so we need to apply it
            action = last_move.action
            if action.get("position") and action.get("action_type") == "place":
                pos = Position(action["position"][0], action["position"][1])
                try:
                    board.place_chip(pos, TeamId(last_move.team))
                except ValueError:
                    pass
            self._live_canvas.update_board(board, last_action=action)

        winner_str = f"Team {record.winner}" if record.winner is not None else "Draw"
        info = (
            f"Game: {record.game_id}\n"
            f"Agents: {', '.join(record.agent_names)}\n"
            f"Winner: {winner_str}\n"
            f"Total turns: {record.total_turns}\n"
            f"Duration: {record.duration_ms:.0f}ms"
        )
        self._live_info.configure(text=info)

        # Also load into replay tab
        self._replay_view.load_record(record)

    def _on_live_game_error(self, error_msg: str) -> None:
        self._live_playing = False
        self._start_btn.configure(state=tk.NORMAL)
        self._live_info.configure(text=f"Error: {error_msg}")
        messagebox.showerror("Game Error", error_msg)

    # ---- Replay Tab ----

    def _load_jsonl(self) -> None:
        path = filedialog.askopenfilename(
            title="Open Game Dataset",
            filetypes=[("JSONL files", "*.jsonl"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            reader = DatasetReader(path)
            records = reader.read_all()
            if not records:
                messagebox.showinfo("Info", "No game records found in file")
                return
            # Load the first record
            self._current_record = records[0]
            self._replay_view.load_record(records[0])
            self._notebook.select(self._replay_view)

            if len(records) > 1:
                messagebox.showinfo(
                    "Info",
                    f"Loaded {len(records)} games. Showing the first one.\n"
                    f"Game ID: {records[0].game_id}",
                )
        except Exception as e:
            messagebox.showerror("Load Error", str(e))

    # ---- Analysis Tab ----

    def _build_analysis_tab(self) -> None:
        frame = self._analysis_frame

        # Board canvas with heatmap
        self._analysis_canvas = BoardCanvas(frame)
        self._analysis_canvas.pack(side=tk.LEFT, padx=5, pady=5)

        # Sidebar
        sidebar = tk.Frame(frame)
        sidebar.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Scoring weights selection
        weights_frame = tk.LabelFrame(sidebar, text="Scoring Weights", padx=5, pady=5)
        weights_frame.pack(fill=tk.X, pady=(0, 5))

        self._weights_var = tk.StringVar(value="Balanced")
        for name in ["Balanced", "Defensive", "Offensive"]:
            tk.Radiobutton(
                weights_frame, text=name, variable=self._weights_var, value=name
            ).pack(anchor="w")

        # Team selection
        team_frame = tk.LabelFrame(sidebar, text="Analyze for Team", padx=5, pady=5)
        team_frame.pack(fill=tk.X, pady=(0, 5))

        self._analysis_team_var = tk.StringVar(value="0")
        for t in range(2):
            tk.Radiobutton(
                team_frame, text=f"Team {t}", variable=self._analysis_team_var, value=str(t)
            ).pack(anchor="w")

        # Compute button
        tk.Button(
            sidebar, text="Compute Heatmap", command=self._compute_heatmap
        ).pack(fill=tk.X, pady=5)

        tk.Button(
            sidebar, text="Clear Heatmap", command=self._clear_heatmap
        ).pack(fill=tk.X, pady=5)

        # Info
        self._analysis_info = tk.Label(
            sidebar, text="Load a game first, then compute heatmap\nfor the current board state.",
            justify=tk.LEFT, anchor="nw", font=("Courier", 9),
        )
        self._analysis_info.pack(fill=tk.BOTH, expand=True, pady=5)

    def _compute_heatmap(self) -> None:
        if self._current_record is None or not self._current_record.moves:
            messagebox.showinfo("Info", "Load a game first (via Live or Replay)")
            return

        weights_name = self._weights_var.get()
        if weights_name == "Balanced":
            weights = BALANCED_WEIGHTS
        elif weights_name == "Defensive":
            weights = DEFENSIVE_WEIGHTS
        else:
            weights = OFFENSIVE_WEIGHTS

        scoring_fn = ScoringFunction(weights)
        team = TeamId(int(self._analysis_team_var.get()))

        # Use the replay's current turn to get board state
        replay_turn = self._replay_view._turn_index
        record = self._current_record
        move = record.moves[min(replay_turn, len(record.moves) - 1)]

        # Reconstruct a minimal game state for scoring
        from ..core.deck import Deck
        from ..core.game_state import GameState

        board = Board.from_list(move.board_snapshot)
        # Create a minimal state for the heatmap computation
        hands: dict[int, list[Any]] = {t: [] for t in range(record.config.get("num_teams", 2))}
        deck = Deck(seed=0)
        state = GameState(
            board=board,
            hands=hands,
            deck=deck,
            current_team=team,
            num_teams=record.config.get("num_teams", 2),
        )

        heatmap = compute_scoring_heatmap(state, team, scoring_fn)
        heatmap_list = heatmap.tolist()

        self._analysis_canvas.set_heatmap(heatmap_list)
        self._analysis_canvas.update_board(board)

        self._analysis_info.configure(
            text=f"Heatmap for Team {team.value}\n"
            f"Weights: {weights_name}\n"
            f"Turn: {move.turn}\n"
            f"Green = high score\n"
            f"Red = low score"
        )

    def _clear_heatmap(self) -> None:
        self._analysis_canvas.clear_heatmap()
        if self._current_record and self._current_record.moves:
            replay_turn = self._replay_view._turn_index
            move = self._current_record.moves[min(replay_turn, len(self._current_record.moves) - 1)]
            board = Board.from_list(move.board_snapshot)
            self._analysis_canvas.update_board(board)
        self._analysis_info.configure(text="Heatmap cleared.")


def main() -> None:
    """Launch the Sequence GUI application."""
    app = SequenceApp()
    app.mainloop()
