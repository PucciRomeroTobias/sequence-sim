"""Scoring function for evaluating Sequence game states."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from typing import TYPE_CHECKING

import numpy as np

from .features import NUM_FEATURES, extract_features

if TYPE_CHECKING:
    from ..core.actions import Action
    from ..core.game_state import GameState
    from ..core.types import TeamId

FEATURE_NAMES: list[str] = [
    "completed_sequences",
    "four_in_a_row",
    "three_in_a_row",
    "two_in_a_row",
    "opp_completed_sequences",
    "opp_four_in_a_row",
    "opp_three_in_a_row",
    "chips_on_board",
    "opp_chips_on_board",
    "center_control",
    "corner_adjacency",
    "hand_pairs",
    "two_eyed_jacks_in_hand",
    "one_eyed_jacks_in_hand",
    "dead_cards_in_hand",
    "shared_line_potential",
    "blocked_lines",
]

assert len(FEATURE_NAMES) == NUM_FEATURES


@dataclass
class ScoringWeights:
    """Weights for the 17 scoring features."""

    completed_sequences: float = 0.0
    four_in_a_row: float = 0.0
    three_in_a_row: float = 0.0
    two_in_a_row: float = 0.0
    opp_completed_sequences: float = 0.0
    opp_four_in_a_row: float = 0.0
    opp_three_in_a_row: float = 0.0
    chips_on_board: float = 0.0
    opp_chips_on_board: float = 0.0
    center_control: float = 0.0
    corner_adjacency: float = 0.0
    hand_pairs: float = 0.0
    two_eyed_jacks_in_hand: float = 0.0
    one_eyed_jacks_in_hand: float = 0.0
    dead_cards_in_hand: float = 0.0
    shared_line_potential: float = 0.0
    blocked_lines: float = 0.0

    def to_array(self) -> np.ndarray:
        return np.array(
            [getattr(self, f.name) for f in fields(self)], dtype=np.float64
        )

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, d: dict[str, float]) -> ScoringWeights:
        return cls(**d)

    @classmethod
    def from_json(cls, s: str) -> ScoringWeights:
        return cls(**json.loads(s))

    @classmethod
    def from_array(cls, arr: np.ndarray) -> ScoringWeights:
        names = [f.name for f in fields(cls)]
        return cls(**{name: float(arr[i]) for i, name in enumerate(names)})


class ScoringFunction:
    """Evaluates game states using weighted feature extraction."""

    __slots__ = ("weights", "_weight_array")

    def __init__(self, weights: ScoringWeights) -> None:
        self.weights = weights
        self._weight_array = weights.to_array()

    def evaluate(self, state: GameState, team: TeamId) -> float:
        """Compute a scalar score for the given team's position."""
        features = extract_features(state, team)
        return float(np.dot(self._weight_array, features))

    def rank_actions(
        self, state: GameState, legal_actions: list[Action], team: TeamId
    ) -> list[tuple[Action, float]]:
        """Rank legal actions by resulting state score (descending).

        Returns list of (action, score) tuples sorted best-first.
        """
        scored: list[tuple[Action, float]] = []
        for action in legal_actions:
            next_state = state.apply_action(action)
            score = self.evaluate(next_state, team)
            scored.append((action, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored


# --- Pre-built weight sets ---

BALANCED_WEIGHTS = ScoringWeights(
    completed_sequences=100.0,
    four_in_a_row=15.0,
    three_in_a_row=5.0,
    two_in_a_row=1.0,
    opp_completed_sequences=-100.0,
    opp_four_in_a_row=-12.0,
    opp_three_in_a_row=-4.0,
    chips_on_board=0.5,
    opp_chips_on_board=-0.5,
    center_control=1.0,
    corner_adjacency=1.5,
    hand_pairs=0.5,
    two_eyed_jacks_in_hand=2.0,
    one_eyed_jacks_in_hand=1.5,
    dead_cards_in_hand=-2.0,
    shared_line_potential=1.0,
    blocked_lines=-0.5,
)

DEFENSIVE_WEIGHTS = ScoringWeights(
    completed_sequences=100.0,
    four_in_a_row=10.0,
    three_in_a_row=3.0,
    two_in_a_row=0.5,
    opp_completed_sequences=-150.0,
    opp_four_in_a_row=-20.0,
    opp_three_in_a_row=-8.0,
    chips_on_board=0.3,
    opp_chips_on_board=-1.0,
    center_control=0.5,
    corner_adjacency=1.0,
    hand_pairs=0.3,
    two_eyed_jacks_in_hand=1.5,
    one_eyed_jacks_in_hand=3.0,
    dead_cards_in_hand=-2.0,
    shared_line_potential=0.5,
    blocked_lines=-0.3,
)

OFFENSIVE_WEIGHTS = ScoringWeights(
    completed_sequences=100.0,
    four_in_a_row=20.0,
    three_in_a_row=8.0,
    two_in_a_row=2.0,
    opp_completed_sequences=-80.0,
    opp_four_in_a_row=-8.0,
    opp_three_in_a_row=-2.0,
    chips_on_board=1.0,
    opp_chips_on_board=-0.3,
    center_control=1.5,
    corner_adjacency=2.0,
    hand_pairs=1.0,
    two_eyed_jacks_in_hand=3.0,
    one_eyed_jacks_in_hand=1.0,
    dead_cards_in_hand=-1.5,
    shared_line_potential=2.0,
    blocked_lines=-0.5,
)
