"""Offensive scorer agent with offensive weights."""

from __future__ import annotations

from .scorer_agent import ScorerAgent
from ..scoring.scoring_function import OFFENSIVE_WEIGHTS


class OffensiveAgent(ScorerAgent):
    """Agent that prioritizes building own sequences aggressively."""

    def __init__(self) -> None:
        super().__init__(weights=OFFENSIVE_WEIGHTS)
