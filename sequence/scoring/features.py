"""Feature extraction from GameState for scoring."""

from __future__ import annotations

import numpy as np

from ..core.board import ALL_LINES, CARD_TO_POSITIONS
from ..core.types import CORNER, CORNERS, EMPTY, Position, TeamId

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.game_state import GameState

NUM_FEATURES = 17


def extract_features(state: GameState, team: TeamId) -> np.ndarray:
    """Extract 17 features from the game state for a given team.

    Features (in order):
     0: completed_sequences
     1: four_in_a_row
     2: three_in_a_row
     3: two_in_a_row
     4: opp_completed_sequences
     5: opp_four_in_a_row
     6: opp_three_in_a_row
     7: chips_on_board
     8: opp_chips_on_board
     9: center_control (own chips in rows/cols 2-7)
    10: corner_adjacency
    11: hand_pairs
    12: two_eyed_jacks_in_hand
    13: one_eyed_jacks_in_hand
    14: dead_cards_in_hand
    15: shared_line_potential
    16: blocked_lines
    """
    board = state.board
    chips = board.chips
    team_val = team.value
    num_teams = state.num_teams

    # Determine opponent team values
    opp_vals: set[int] = set()
    for t in range(num_teams):
        if t != team_val:
            opp_vals.add(t)

    # --- Line scanning ---
    # For each line of 5 positions, count own chips, opponent chips, corners
    completed_sequences = 0
    four_in_a_row = 0
    three_in_a_row = 0
    two_in_a_row = 0
    opp_completed_sequences = 0
    opp_four_in_a_row = 0
    opp_three_in_a_row = 0
    shared_line_potential = 0
    blocked_lines = 0

    for line in ALL_LINES:
        own_count = 0
        opp_count = 0
        corner_count = 0
        for pos in line:
            val = int(chips[pos.row, pos.col])
            if val == team_val:
                own_count += 1
            elif val in opp_vals:
                opp_count += 1
            elif val == CORNER:
                corner_count += 1

        own_total = own_count + corner_count
        opp_total = opp_count + corner_count

        # Own line analysis: own chips + corners, no opponent chips
        if opp_count == 0:
            if own_total >= 5:
                completed_sequences += 1
            elif own_total == 4:
                four_in_a_row += 1
            elif own_total == 3:
                three_in_a_row += 1
            elif own_total == 2:
                two_in_a_row += 1

        # Opponent line analysis: opp chips + corners, no own chips
        if own_count == 0:
            if opp_total >= 5:
                opp_completed_sequences += 1
            elif opp_total == 4:
                opp_four_in_a_row += 1
            elif opp_total == 3:
                opp_three_in_a_row += 1

        # Shared line potential: lines where BOTH teams have chips (contested)
        if own_count > 0 and opp_count > 0:
            blocked_lines += 1

        # Lines with own chips and empty slots (no opponent) = potential
        if own_count >= 1 and opp_count == 0:
            shared_line_potential += 1

    # --- Chip counts ---
    chips_on_board = 0
    opp_chips_on_board = 0
    center_control = 0
    corner_adjacency = 0

    # Adjacent positions to corners
    corner_adj_positions: set[Position] = set()
    for cp in CORNERS:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = cp.row + dr, cp.col + dc
                if 0 <= nr < 10 and 0 <= nc < 10:
                    corner_adj_positions.add(Position(nr, nc))

    for r in range(10):
        for c in range(10):
            val = int(chips[r, c])
            if val == team_val:
                chips_on_board += 1
                if 2 <= r <= 7 and 2 <= c <= 7:
                    center_control += 1
                if Position(r, c) in corner_adj_positions:
                    corner_adjacency += 1
            elif val in opp_vals:
                opp_chips_on_board += 1

    # --- Hand analysis ---
    hand = state.hands.get(team_val, [])
    hand_pairs = 0
    two_eyed_jacks_in_hand = 0
    one_eyed_jacks_in_hand = 0
    dead_cards_in_hand = 0

    # Count card occurrences for pairs
    card_counts: dict[object, int] = {}
    for card in hand:
        if card.is_two_eyed_jack:
            two_eyed_jacks_in_hand += 1
        elif card.is_one_eyed_jack:
            one_eyed_jacks_in_hand += 1
        else:
            card_counts[card] = card_counts.get(card, 0) + 1
            # Check if dead card
            positions = CARD_TO_POSITIONS.get(card, [])
            if all(not board.is_empty(p) for p in positions):
                dead_cards_in_hand += 1

    for count in card_counts.values():
        if count >= 2:
            hand_pairs += 1

    features = np.array(
        [
            completed_sequences,       # 0
            four_in_a_row,              # 1
            three_in_a_row,             # 2
            two_in_a_row,               # 3
            opp_completed_sequences,    # 4
            opp_four_in_a_row,          # 5
            opp_three_in_a_row,         # 6
            chips_on_board,             # 7
            opp_chips_on_board,         # 8
            center_control,             # 9
            corner_adjacency,           # 10
            hand_pairs,                 # 11
            two_eyed_jacks_in_hand,     # 12
            one_eyed_jacks_in_hand,     # 13
            dead_cards_in_hand,         # 14
            shared_line_potential,      # 15
            blocked_lines,              # 16
        ],
        dtype=np.float64,
    )
    return features
