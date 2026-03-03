"""Scoring function for evaluating Sequence game states."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from typing import TYPE_CHECKING

import numpy as np

from .features import NUM_FEATURES, extract_features

if TYPE_CHECKING:
    from ..core.actions import Action
    from ..core.card_tracker import CardTracker
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
    # Card-tracking features
    "guaranteed_positions",
    "viable_four_in_a_row",
    "viable_three_in_a_row",
    "opp_blockable_four",
    "opp_unblockable_four",
    "fork_count",
    "dead_positions_for_us",
    "dead_lines",
    "opp_dead_lines",
    "card_monopoly_count",
    "sequence_proximity",
    # Advanced strategy features
    "position_line_score",
    "anchor_overlap_count",
    "chip_clustering",
    "early_jack_usage_penalty",
    "jack_save_value",
]

assert len(FEATURE_NAMES) == NUM_FEATURES


@dataclass
class ScoringWeights:
    """Weights for the 33 scoring features."""

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
    # Card-tracking features (default 0.0 for backward compat)
    guaranteed_positions: float = 0.0
    viable_four_in_a_row: float = 0.0
    viable_three_in_a_row: float = 0.0
    opp_blockable_four: float = 0.0
    opp_unblockable_four: float = 0.0
    fork_count: float = 0.0
    dead_positions_for_us: float = 0.0
    dead_lines: float = 0.0
    opp_dead_lines: float = 0.0
    card_monopoly_count: float = 0.0
    sequence_proximity: float = 0.0
    # Advanced strategy features (default 0.0 for backward compat)
    position_line_score: float = 0.0
    anchor_overlap_count: float = 0.0
    chip_clustering: float = 0.0
    early_jack_usage_penalty: float = 0.0
    jack_save_value: float = 0.0

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
        # Filter to only known field names for backward compatibility
        known = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in d.items() if k in known}
        return cls(**filtered)

    @classmethod
    def from_json(cls, s: str) -> ScoringWeights:
        return cls.from_dict(json.loads(s))

    @classmethod
    def from_array(cls, arr: np.ndarray) -> ScoringWeights:
        names = [f.name for f in fields(cls)]
        # Handle arrays shorter than current field count (backward compat)
        d: dict[str, float] = {}
        for i, name in enumerate(names):
            if i < len(arr):
                d[name] = float(arr[i])
            else:
                d[name] = 0.0
        return cls(**d)


class ScoringFunction:
    """Evaluates game states using weighted feature extraction."""

    __slots__ = ("weights", "_weight_array")

    def __init__(self, weights: ScoringWeights) -> None:
        self.weights = weights
        self._weight_array = weights.to_array()

    def evaluate(
        self,
        state: GameState,
        team: TeamId,
        tracker: CardTracker | None = None,
    ) -> float:
        """Compute a scalar score for the given team's position."""
        features = extract_features(state, team, tracker=tracker)
        return float(np.dot(self._weight_array, features))

    def rank_actions(
        self,
        state: GameState,
        legal_actions: list[Action],
        team: TeamId,
        tracker: CardTracker | None = None,
    ) -> list[tuple[Action, float]]:
        """Rank legal actions by resulting state score (descending).

        Returns list of (action, score) tuples sorted best-first.
        """
        scored: list[tuple[Action, float]] = []
        for action in legal_actions:
            next_state = state.apply_action(action)
            score = self.evaluate(next_state, team, tracker=tracker)
            scored.append((action, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def rank_actions_fast(
        self,
        state: GameState,
        legal_actions: list[Action],
        team: TeamId,
        tracker: CardTracker | None = None,
    ) -> list[tuple[Action, float]]:
        """Rank legal actions using virtual apply for PLACE actions.

        For PLACE actions (~90% of actions), mutates board/hand in-place
        temporarily and reverts, avoiding expensive deep copies.
        For REMOVE and DEAD_CARD_DISCARD, falls back to full apply_action.

        Returns list of (action, score) tuples sorted best-first.
        """
        from ..core.actions import ActionType
        from ..core.types import EMPTY

        board = state.board
        chips = board.chips
        hand = state.hands.get(team.value, [])
        team_val = team.value
        weight_array = self._weight_array
        scored: list[tuple[Action, float]] = []

        for action in legal_actions:
            if action.action_type == ActionType.PLACE and action.position is not None:
                pos = action.position
                r, c = pos.row, pos.col
                # --- Virtual apply: mutate in-place ---
                old_chip = int(chips[r, c])
                chips[r, c] = team_val
                board.empty_positions.discard(pos)
                # Temporarily remove card from hand
                hand.remove(action.card)

                # Increment turn_number for feature extraction
                state.turn_number += 1
                features = extract_features(state, team, tracker=tracker)
                score = float(np.dot(weight_array, features))
                state.turn_number -= 1

                # --- Revert ---
                hand.append(action.card)
                chips[r, c] = old_chip
                if old_chip == EMPTY:
                    board.empty_positions.add(pos)

                scored.append((action, score))
            else:
                # Fallback for REMOVE and DEAD_CARD_DISCARD
                next_state = state.apply_action(action)
                score = self.evaluate(next_state, team, tracker=tracker)
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
    guaranteed_positions=3.0,
    viable_four_in_a_row=18.0,
    viable_three_in_a_row=6.0,
    opp_blockable_four=-15.0,
    opp_unblockable_four=-5.0,
    fork_count=8.0,
    dead_positions_for_us=-1.0,
    dead_lines=-2.0,
    opp_dead_lines=1.5,
    card_monopoly_count=2.0,
    sequence_proximity=4.0,
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
    guaranteed_positions=2.0,
    viable_four_in_a_row=12.0,
    viable_three_in_a_row=4.0,
    opp_blockable_four=-25.0,
    opp_unblockable_four=-8.0,
    fork_count=5.0,
    dead_positions_for_us=-1.5,
    dead_lines=-3.0,
    opp_dead_lines=2.0,
    card_monopoly_count=1.5,
    sequence_proximity=3.0,
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
    guaranteed_positions=4.0,
    viable_four_in_a_row=22.0,
    viable_three_in_a_row=9.0,
    opp_blockable_four=-10.0,
    opp_unblockable_four=-3.0,
    fork_count=12.0,
    dead_positions_for_us=-0.5,
    dead_lines=-1.5,
    opp_dead_lines=1.0,
    card_monopoly_count=3.0,
    sequence_proximity=6.0,
)

# Smart weights: designed to work with CardTracker-enabled agents
SMART_WEIGHTS = ScoringWeights(
    completed_sequences=91.415,
    four_in_a_row=16.0,
    three_in_a_row=5.0,
    two_in_a_row=1.426,
    opp_completed_sequences=-100.0,
    opp_four_in_a_row=-11.677,
    opp_three_in_a_row=-5.0,
    chips_on_board=0.4,
    opp_chips_on_board=-0.4,
    center_control=1.604,
    corner_adjacency=1.5,
    hand_pairs=0.859,
    two_eyed_jacks_in_hand=2.5,
    one_eyed_jacks_in_hand=2.0,
    dead_cards_in_hand=-2.549,
    shared_line_potential=0.553,
    blocked_lines=-0.95,
    guaranteed_positions=4.0,
    viable_four_in_a_row=30.684,
    viable_three_in_a_row=6.733,
    opp_blockable_four=-22.0,
    opp_unblockable_four=-5.71,
    fork_count=12.419,
    dead_positions_for_us=-1.5,
    dead_lines=-4.689,
    opp_dead_lines=2.307,
    card_monopoly_count=3.0,
    sequence_proximity=4.465,
    position_line_score=2.0,
    anchor_overlap_count=6.0,
    chip_clustering=3.551,
    early_jack_usage_penalty=-5.0,
    jack_save_value=4.0,
)
