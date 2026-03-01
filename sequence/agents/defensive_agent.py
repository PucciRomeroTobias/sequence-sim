"""Defensive scorer agent with defensive weights."""

from __future__ import annotations

from .scorer_agent import ScorerAgent
from ..scoring.scoring_function import DEFENSIVE_WEIGHTS


class DefensiveAgent(ScorerAgent):
    """Agent that prioritizes blocking opponent sequences."""

    def __init__(self) -> None:
        super().__init__(weights=DEFENSIVE_WEIGHTS)
