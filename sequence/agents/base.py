"""Base agent interface for the Sequence game."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.actions import Action
    from ..core.game import GameConfig
    from ..core.game_state import GameState
    from ..core.types import TeamId


class Agent(ABC):
    """Abstract base class for Sequence game agents."""

    @abstractmethod
    def choose_action(
        self, state: GameState, legal_actions: list[Action]
    ) -> Action:
        """Choose an action from the list of legal actions.

        Args:
            state: Current game state (visible to this agent).
            legal_actions: All legal actions available.

        Returns:
            The chosen action.
        """
        ...

    def notify_game_start(self, team: TeamId, config: GameConfig) -> None:
        """Called at the start of a game."""
        pass

    def notify_action(self, action: Action, team: TeamId) -> None:
        """Called after any player takes an action."""
        pass
