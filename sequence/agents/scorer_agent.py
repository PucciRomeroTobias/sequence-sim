"""Scorer agent that uses a ScoringFunction to choose actions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import Agent
from ..scoring.scoring_function import BALANCED_WEIGHTS, ScoringFunction, ScoringWeights

if TYPE_CHECKING:
    from ..core.actions import Action
    from ..core.game import GameConfig
    from ..core.game_state import GameState
    from ..core.types import TeamId


class ScorerAgent(Agent):
    """Agent that evaluates all legal actions using a ScoringFunction and picks the best."""

    def __init__(self, weights: ScoringWeights | None = None) -> None:
        self._weights = weights or BALANCED_WEIGHTS
        self._scoring = ScoringFunction(self._weights)
        self._team: TeamId | None = None

    def notify_game_start(self, team: TeamId, config: GameConfig) -> None:
        self._team = team

    def choose_action(self, state: GameState, legal_actions: list[Action]) -> Action:
        team = self._team if self._team is not None else state.current_team
        ranked = self._scoring.rank_actions(state, legal_actions, team)
        return ranked[0][0]
