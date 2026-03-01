"""Tournament runner with parallel execution and side-swapping."""

from __future__ import annotations

import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable

from ..core.game import GameConfig, GameRecord
from .runner import run_single_game

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore[assignment]


@dataclass
class TournamentResult:
    """Aggregated results of a tournament."""

    records: list[GameRecord] = field(default_factory=list)
    agent_names: list[str] = field(default_factory=list)

    @property
    def total_games(self) -> int:
        return len(self.records)

    @property
    def wins(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for name in self.agent_names:
            counts[name] = 0
        for rec in self.records:
            if rec.winner is not None:
                winner_name = rec.agent_names[rec.winner]
                counts[winner_name] = counts.get(winner_name, 0) + 1
        return counts

    @property
    def draws(self) -> int:
        return sum(1 for r in self.records if r.winner is None)

    @property
    def win_rates(self) -> dict[str, float]:
        if self.total_games == 0:
            return {}
        w = self.wins
        return {name: count / self.total_games for name, count in w.items()}


def _run_game_wrapper(args: tuple[list[Callable], GameConfig | None]) -> GameRecord:
    """Wrapper for ProcessPoolExecutor that unpacks args."""
    factories, config = args
    return run_single_game(factories, config)


class Tournament:
    """Run multiple games between agents, optionally swapping sides."""

    def __init__(
        self,
        agent_factories: list[Callable[[], Any]],
        num_games: int = 100,
        config: GameConfig | None = None,
        swap_sides: bool = True,
        max_workers: int | None = None,
        show_progress: bool = True,
    ) -> None:
        self.agent_factories = agent_factories
        self.num_games = num_games
        self.config = config or GameConfig()
        self.swap_sides = swap_sides
        self.max_workers = max_workers
        self.show_progress = show_progress

    def run(self) -> TournamentResult:
        """Run the tournament and return aggregated results."""
        tasks: list[tuple[list[Callable], GameConfig | None]] = []

        if self.swap_sides:
            # Half normal, half swapped
            half = self.num_games // 2
            remainder = self.num_games - 2 * half
            for i in range(half):
                cfg = GameConfig(
                    num_teams=self.config.num_teams,
                    sequences_to_win=self.config.sequences_to_win,
                    seed=i if self.config.seed is None else self.config.seed + i,
                    max_turns=self.config.max_turns,
                )
                tasks.append((list(self.agent_factories), cfg))
            for i in range(half + remainder):
                cfg = GameConfig(
                    num_teams=self.config.num_teams,
                    sequences_to_win=self.config.sequences_to_win,
                    seed=half + i if self.config.seed is None else self.config.seed + half + i,
                    max_turns=self.config.max_turns,
                )
                swapped = list(reversed(self.agent_factories))
                tasks.append((swapped, cfg))
        else:
            for i in range(self.num_games):
                cfg = GameConfig(
                    num_teams=self.config.num_teams,
                    sequences_to_win=self.config.sequences_to_win,
                    seed=i if self.config.seed is None else self.config.seed + i,
                    max_turns=self.config.max_turns,
                )
                tasks.append((list(self.agent_factories), cfg))

        records: list[GameRecord] = []
        use_progress = self.show_progress and tqdm is not None

        if self.max_workers == 1:
            # Sequential execution (useful for debugging)
            iterator = tasks
            if use_progress:
                iterator = tqdm(tasks, desc="Games", unit="game")  # type: ignore[assignment]
            for task in iterator:
                records.append(_run_game_wrapper(task))
        else:
            with ProcessPoolExecutor(max_workers=self.max_workers) as pool:
                futures = [pool.submit(_run_game_wrapper, t) for t in tasks]
                if use_progress:
                    pbar = tqdm(total=len(futures), desc="Games", unit="game")  # type: ignore[assignment]
                for future in as_completed(futures):
                    records.append(future.result())
                    if use_progress:
                        pbar.update(1)  # type: ignore[possibly-undefined]
                if use_progress:
                    pbar.close()  # type: ignore[possibly-undefined]

        # Determine canonical agent names from first record
        agent_names: list[str] = []
        if records:
            # Collect all unique agent names across all records
            seen: set[str] = set()
            for rec in records:
                for name in rec.agent_names:
                    if name not in seen:
                        seen.add(name)
                        agent_names.append(name)

        return TournamentResult(records=records, agent_names=agent_names)
