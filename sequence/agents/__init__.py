"""Sequence game agents."""

from .base import Agent
from .defensive_agent import DefensiveAgent
from .greedy_agent import GreedyAgent
from .offensive_agent import OffensiveAgent
from .random_agent import RandomAgent
from .scorer_agent import ScorerAgent
from .smart_agent import SmartAgent

__all__ = [
    "Agent",
    "DefensiveAgent",
    "GreedyAgent",
    "NeuralAgent",
    "OffensiveAgent",
    "RandomAgent",
    "ScorerAgent",
    "SmartAgent",
]

try:
    from .neural_agent import NeuralAgent
except ImportError:
    pass
