"""Feature extraction from GameState for scoring."""

from __future__ import annotations

import numpy as np

from ..core.board import ALL_LINES, CARD_TO_POSITIONS, LAYOUT, POSITION_TO_LINES
from ..core.types import CORNER, CORNERS, EMPTY, Position, TeamId

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.card_tracker import CardTracker
    from ..core.game_state import GameState

NUM_FEATURES = 35


def extract_features(
    state: GameState, team: TeamId, tracker: CardTracker | None = None
) -> np.ndarray:
    """Extract 35 features from the game state for a given team.

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
    --- Card-tracking features (require tracker, default to 0) ---
    17: guaranteed_positions
    18: viable_four_in_a_row
    19: viable_three_in_a_row
    20: opp_blockable_four
    21: opp_unblockable_four
    22: fork_count
    23: dead_positions_for_us
    24: dead_lines
    25: opp_dead_lines
    26: card_monopoly_count
    27: sequence_proximity
    --- Advanced strategy features ---
    28: position_line_score
    29: anchor_overlap_count
    30: chip_clustering
    31: early_jack_usage_penalty
    32: jack_save_value
    33: viable_line_connectivity
    34: weighted_blocking
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
    viable_line_connectivity = 0
    weighted_blocking = 0

    # Card-tracking enhanced line features
    viable_four_in_a_row = 0
    viable_three_in_a_row = 0
    opp_blockable_four = 0
    opp_unblockable_four = 0
    dead_lines = 0
    opp_dead_lines = 0
    sequence_proximity = 0.0

    hand = state.hands.get(team_val, [])

    # Pre-compute which cards we hold (for viability/blockability checks)
    hand_card_set: set = set(hand) if tracker else set()

    for line in ALL_LINES:
        own_count = 0
        opp_count = 0
        corner_count = 0
        empty_positions: list[Position] = []
        for pos in line:
            val = int(chips[pos.row, pos.col])
            if val == team_val:
                own_count += 1
            elif val in opp_vals:
                opp_count += 1
            elif val == CORNER:
                corner_count += 1
            elif val == EMPTY:
                empty_positions.append(pos)

        own_total = own_count + corner_count
        opp_total = opp_count + corner_count

        # Own line analysis: own chips + corners, no opponent chips
        if opp_count == 0:
            if own_total >= 5:
                completed_sequences += 1
            elif own_total == 4:
                four_in_a_row += 1
                if tracker and empty_positions:
                    if _is_position_fillable(empty_positions[0], tracker, hand):
                        viable_four_in_a_row += 1
                # Sequence proximity: 4/5 = 0.8 contribution
                sequence_proximity += 0.8
            elif own_total == 3:
                three_in_a_row += 1
                if tracker:
                    fillable = sum(
                        1 for p in empty_positions
                        if _is_position_fillable(p, tracker, hand)
                    )
                    if fillable == len(empty_positions):
                        viable_three_in_a_row += 1
                # Sequence proximity: 3/5 = 0.6 contribution
                sequence_proximity += 0.3
            elif own_total == 2:
                two_in_a_row += 1

        # Opponent line analysis: opp chips + corners, no own chips
        if own_count == 0:
            if opp_total >= 5:
                opp_completed_sequences += 1
            elif opp_total == 4:
                opp_four_in_a_row += 1
                if tracker and empty_positions:
                    # Can we block the remaining empty position?
                    block_pos = empty_positions[0]
                    if _can_we_place_at(block_pos, team_val, hand, chips):
                        opp_blockable_four += 1
                    else:
                        opp_unblockable_four += 1
            elif opp_total == 3:
                opp_three_in_a_row += 1

        # Blocked lines
        if own_count > 0 and opp_count > 0:
            blocked_lines += 1

        # Shared line potential
        if own_count >= 1 and opp_count == 0:
            shared_line_potential += 1

        # Viable line connectivity: lines without opponent, 2+ own chips
        if opp_count == 0 and own_total >= 2:
            viable_line_connectivity += own_total - 1

        # Weighted blocking: contested lines, weighted by opponent progress
        if own_count > 0 and opp_count > 0:
            weighted_blocking += opp_count

        # Dead line detection (requires tracker)
        if tracker:
            # A line is dead for us if: no opponent chips, but some empty positions
            # are permanently unfillable
            if opp_count == 0 and own_count > 0 and empty_positions:
                all_fillable = all(
                    _is_position_fillable(p, tracker, hand)
                    for p in empty_positions
                )
                if not all_fillable:
                    dead_lines += 1

            # A line is dead for opponent if: no own chips, but some empty positions
            # are permanently unfillable (by anyone)
            if own_count == 0 and opp_count > 0 and empty_positions:
                any_dead = any(
                    tracker.is_position_permanently_dead(p, chips)
                    for p in empty_positions
                )
                if any_dead:
                    opp_dead_lines += 1

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

    # --- Card-tracking features ---
    guaranteed_positions = 0
    fork_count = 0
    dead_positions_for_us = 0
    card_monopoly_count = 0

    if tracker:
        # Guaranteed positions
        guaranteed_set = tracker.get_guaranteed_positions(hand, chips)
        guaranteed_positions = len(guaranteed_set)

        # Card monopoly count: cards where we hold all remaining copies
        seen_cards: set = set()
        for card in hand:
            if card.is_jack or card in seen_cards:
                continue
            seen_cards.add(card)
            if tracker.copies_remaining_in_pool(card) == 0:
                card_monopoly_count += 1

        # Dead positions for us: empty positions we can never fill
        for r in range(10):
            for c in range(10):
                pos = Position(r, c)
                if int(chips[r, c]) == EMPTY and pos not in CORNERS:
                    if tracker.is_position_permanently_dead(pos, chips):
                        dead_positions_for_us += 1

        # Fork detection: positions that would advance 2+ of our lines
        # A "fork" is an empty position where placing a chip extends
        # 2 or more lines that already have 3+ own chips
        for r in range(10):
            for c in range(10):
                pos = Position(r, c)
                if int(chips[r, c]) != EMPTY:
                    continue
                if pos in CORNERS:
                    continue
                lines_advanced = 0
                line_indices = POSITION_TO_LINES.get(pos, [])
                for line_idx in line_indices:
                    line = ALL_LINES[line_idx]
                    own_in_line = 0
                    opp_in_line = 0
                    for lp in line:
                        v = int(chips[lp.row, lp.col])
                        if v == team_val or v == CORNER:
                            own_in_line += 1
                        elif v in opp_vals:
                            opp_in_line += 1
                    if opp_in_line == 0 and own_in_line >= 3:
                        lines_advanced += 1
                if lines_advanced >= 2:
                    fork_count += 1

    # --- Advanced strategy features ---

    # 28: position_line_score — sum of line counts for own chips (normalized by 12)
    position_line_score = 0.0
    for r in range(10):
        for c in range(10):
            if int(chips[r, c]) == team_val:
                pos = Position(r, c)
                position_line_score += len(POSITION_TO_LINES.get(pos, [])) / 12.0

    # 29: anchor_overlap_count — chips in completed sequences that also belong
    # to at least 1 incomplete line without opponent chips (reusable anchors)
    anchor_overlap_count = 0
    completed_seqs = board.get_all_sequences(team)
    if completed_seqs:
        seq_positions: set[Position] = set()
        for seq in completed_seqs:
            seq_positions |= seq
        for pos in seq_positions:
            line_indices = POSITION_TO_LINES.get(pos, [])
            for line_idx in line_indices:
                line = ALL_LINES[line_idx]
                # Check if this line is incomplete and has no opponent chips
                own_in_line = 0
                opp_in_line = 0
                for lp in line:
                    v = int(chips[lp.row, lp.col])
                    if v == team_val or v == CORNER:
                        own_in_line += 1
                    elif v in opp_vals:
                        opp_in_line += 1
                if opp_in_line == 0 and own_in_line < 5:
                    anchor_overlap_count += 1
                    break  # Count each position only once

    # 30: chip_clustering — max concentration in a 5x5 quadrant
    quadrant_counts = [0, 0, 0, 0]
    for r in range(10):
        for c in range(10):
            if int(chips[r, c]) == team_val:
                q = (0 if r < 5 else 1) * 2 + (0 if c < 5 else 1)
                quadrant_counts[q] += 1
    total_chips_val = max(chips_on_board, 1)
    chip_clustering = max(quadrant_counts) / total_chips_val

    # 31: early_jack_usage_penalty — penalizes states with fewer jacks early game
    # Differs between actions because using a jack reduces jacks in next_state
    early_jack_usage_penalty = max(0.0, 1.0 - state.turn_number / 50.0) * (
        1.0 / (1.0 + two_eyed_jacks_in_hand + one_eyed_jacks_in_hand)
    )

    # 32: jack_save_value — value of saving jacks (decays over time)
    jack_save_value = (two_eyed_jacks_in_hand + one_eyed_jacks_in_hand) * max(
        0.0, 1.0 - state.turn_number / 50.0
    )

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
            # Card-tracking features
            guaranteed_positions,       # 17
            viable_four_in_a_row,       # 18
            viable_three_in_a_row,      # 19
            opp_blockable_four,         # 20
            opp_unblockable_four,       # 21
            fork_count,                 # 22
            dead_positions_for_us,      # 23
            dead_lines,                 # 24
            opp_dead_lines,             # 25
            card_monopoly_count,        # 26
            sequence_proximity,         # 27
            # Advanced strategy features
            position_line_score,        # 28
            anchor_overlap_count,       # 29
            chip_clustering,            # 30
            early_jack_usage_penalty,   # 31
            jack_save_value,            # 32
            viable_line_connectivity,   # 33
            weighted_blocking,          # 34
        ],
        dtype=np.float64,
    )
    return features


NUM_EXTENDED_FEATURES = 35 + 15  # 35 base + 15 derived


def extract_features_extended(
    state: GameState, team: TeamId, tracker: CardTracker | None = None
) -> np.ndarray:
    """Extract 35 base features + 15 derived interaction features.

    The derived features capture non-linear interactions that a linear
    scoring function cannot represent:
    - Offensive synergies (viable lines × proximity)
    - Defensive urgency (opponent threats × our blocking capacity)
    - Fork value modulated by opponent's removal capability
    - Game phase interactions (early/mid/late adjustments)
    - Spatial synergies (clustering × line potential)

    Returns numpy array of 50 float64 features.
    """
    base = extract_features(state, team, tracker=tracker)

    # Aliases for readability (indices match feature list)
    completed_seq = base[0]
    four_row = base[1]
    three_row = base[2]
    opp_completed = base[4]
    opp_four = base[5]
    opp_three = base[6]
    chips = base[7]
    opp_chips = base[8]
    center = base[9]
    two_eyed_jacks = base[12]
    one_eyed_jacks = base[13]
    dead_cards = base[14]
    shared_line = base[15]
    guaranteed = base[17]
    viable_four = base[18]
    viable_three = base[19]
    opp_blockable = base[20]
    opp_unblockable = base[21]
    fork = base[22]
    dead_lines = base[24]
    monopoly = base[26]
    seq_proximity = base[27]
    pos_line_score = base[28]
    anchor_overlap = base[29]
    clustering = base[30]
    jack_save = base[32]

    # Game phase: 0.0 (start) to 1.0 (late game)
    game_phase = min(state.turn_number / 80.0, 1.0)

    derived = np.array([
        # 35: Offensive completion potential: 4-in-a-row that can actually be completed
        four_row * viable_four,
        # 36: Building momentum: 3-in-a-row with viable completions
        three_row * viable_three,
        # 37: Defensive urgency: opponent threats we can block with jacks
        opp_four * one_eyed_jacks,
        # 38: Unblockable danger: opponent has threats AND we can't block
        opp_four * (1.0 / (1.0 + one_eyed_jacks)),
        # 39: Fork power: forks are deadly when opponent lacks removal
        fork * (1.0 / (1.0 + opp_blockable)),
        # 40: Guaranteed + proximity = close to winning with certainty
        guaranteed * seq_proximity,
        # 41: Race condition: both sides close to completing
        four_row * opp_four,
        # 42: Spatial dominance: clustering + line control
        clustering * pos_line_score,
        # 43: Center + shared lines = positional advantage
        center * shared_line,
        # 44: Dead card burden scales with game phase
        dead_cards * game_phase,
        # 45: Jack value in late game (jacks + forks late = lethal)
        jack_save * fork * game_phase,
        # 46: Monopoly + viable lines = guaranteed progress
        monopoly * viable_three,
        # 47: Anchor reuse potential: completed sequences feeding new lines
        anchor_overlap * three_row,
        # 48: Game phase itself (lets network learn phase-dependent strategies)
        game_phase,
        # 49: Material advantage (chip difference, useful for positional eval)
        chips - opp_chips,
    ], dtype=np.float64)

    return np.concatenate([base, derived])


def _is_position_fillable(
    pos: Position, tracker: CardTracker, hand: list,
) -> bool:
    """Check if a position can potentially be filled (card still exists somewhere)."""
    layout_card = LAYOUT[pos.row][pos.col]
    if layout_card is None:
        return False
    # Check if we hold the card
    for c in hand:
        if c == layout_card:
            return True
        if c.is_two_eyed_jack:
            return True
    # Check if copies remain in the pool
    if tracker.copies_remaining_in_pool(layout_card) > 0:
        return True
    # Check if two-eyed jacks remain
    from ..core.card import Card
    from ..core.types import Rank, Suit
    for suit in (Suit.DIAMONDS, Suit.CLUBS):
        tej = Card(Rank.JACK, suit)
        if tracker.copies_remaining_in_pool(tej) > 0:
            return True
    return False


def _can_we_place_at(
    pos: Position, team_val: int, hand: list, chips,
) -> bool:
    """Check if we can currently place a chip at this position (from our hand)."""
    if int(chips[pos.row, pos.col]) != EMPTY:
        return False
    layout_card = LAYOUT[pos.row][pos.col]
    if layout_card is None:
        return False
    for card in hand:
        if card == layout_card:
            return True
        if card.is_two_eyed_jack:
            return True
    return False
