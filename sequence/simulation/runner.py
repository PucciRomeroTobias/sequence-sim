"""Single-game runner for multiprocessing compatibility."""

from __future__ import annotations

from typing import Any, Callable

from ..core.game import Game, GameConfig, GameRecord


def run_single_game(
    agent_factories: list[Callable[[], Any]],
    config: GameConfig | None = None,
) -> GameRecord:
    """Run a single game and return the record.

    This function is pickle-friendly: it accepts factories (callables)
    rather than agent instances, so it can be dispatched to worker processes.

    Args:
        agent_factories: List of callables that create Agent instances.
        config: Game configuration. Defaults to standard 2-player config.

    Returns:
        A GameRecord with the full game history.
    """
    game = Game(agent_factories=agent_factories, config=config)
    return game.play()
