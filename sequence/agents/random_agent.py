"""Random agent that picks uniformly from legal actions."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

from .base import Agent

if TYPE_CHECKING:
    from ..core.actions import Action
    from ..core.game_state import GameState


class RandomAgent(Agent):
    """Agent that selects a random legal action each turn."""

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def choose_action(self, state: GameState, legal_actions: list[Action]) -> Action:
        return self._rng.choice(legal_actions)
