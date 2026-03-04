"""Extended features for the ExpertAgent (35 base + 12 new = 47 total)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ...core.board import ALL_LINES, CARD_TO_POSITIONS, LAYOUT, POSITION_TO_LINES
from ...core.types import CORNER, CORNERS, EMPTY, Position, TeamId
from ...scoring.features import extract_features

if TYPE_CHECKING:
    from ...core.card_tracker import CardTracker
    from ...core.game_state import GameState

NUM_EXPERT_FEATURES = 47  # 35 base + 12 new


def extract_expert_features(
    state: GameState,
    team: TeamId,
    tracker: CardTracker | None = None,
) -> np.ndarray:
    """Extract 47 features: 35 base + 12 expert-specific.

    New features (indices 35-46):
     35: silent_threat_count
     36: corner_line_progress
     37: phantom_threat_count
     38: opp_threat_reachability
     39: permanent_disruption_potential
     40: hand_flexibility
     41: race_advantage
     42: multi_block_value
     43: key_card_count
     44: urgency_clock
     45: alive_line_position_score
     46: open_ended_three_count
    """
    base = extract_features(state, team, tracker=tracker)

    board = state.board
    chips = board.chips
    team_val = team.value
    hand = state.hands.get(team_val, [])

    opp_vals: set[int] = set()
    for t in range(state.num_teams):
        if t != team_val:
            opp_vals.add(t)

    silent_threat_count = 0
    corner_line_progress = 0
    phantom_threat_count = 0
    opp_threat_reachability = 0.0
    open_ended_three_count = 0

    own_viable_four = 0
    opp_viable_four = 0
    opp_three_reachable_sum = 0.0
    opp_three_count_for_avg = 0

    alive_line_position_score = 0.0

    # Track which positions are in alive lines (no opp chips) for pos score
    alive_line_positions: set[Position] = set()

    # Pre-compute corner positions set for corner line detection
    corner_set = CORNERS

    for line in ALL_LINES:
        own_count = 0
        opp_count = 0
        corner_count = 0
        empty_positions: list[Position] = []
        has_corner_pos = False

        for pos in line:
            val = int(chips[pos.row, pos.col])
            if val == team_val:
                own_count += 1
            elif val in opp_vals:
                opp_count += 1
            elif val == CORNER:
                corner_count += 1
                has_corner_pos = True
            elif val == EMPTY:
                empty_positions.append(pos)
                if pos in corner_set:
                    has_corner_pos = True

        own_total = own_count + corner_count

        # Own lines analysis (no opponent)
        if opp_count == 0:
            # Track alive line positions for position score
            for pos in line:
                alive_line_positions.add(pos)

            if own_total == 4 and len(empty_positions) == 1:
                own_viable_four += 1

            # Silent threat: 3 chips, 2 empty holes, both fillable, no opponent
            if own_total == 3 and len(empty_positions) == 2 and tracker is not None:
                both_fillable = all(
                    _is_fillable(p, tracker, hand) for p in empty_positions
                )
                if both_fillable:
                    silent_threat_count += 1

            # Corner line progress: line passes through a corner, 3+ own chips
            if has_corner_pos and own_total >= 3:
                corner_line_progress += 1

            # Phantom threat: 3+ own chips with a hole we monopolize
            if own_total >= 3 and tracker is not None:
                for ep in empty_positions:
                    layout_card = LAYOUT[ep.row][ep.col]
                    if layout_card is not None:
                        remaining = tracker.copies_remaining_in_pool(layout_card)
                        in_hand = tracker.copies_in_hand(layout_card)
                        if in_hand > 0 and remaining == 0:
                            phantom_threat_count += 1
                            break  # Count line once

            # Open-ended three: 3 chips, both endpoints of the line are empty
            if own_total == 3 and len(empty_positions) >= 2:
                first_pos = line[0]
                last_pos = line[4]
                first_empty = int(chips[first_pos.row, first_pos.col]) == EMPTY
                last_empty = int(chips[last_pos.row, last_pos.col]) == EMPTY
                if first_empty and last_empty:
                    open_ended_three_count += 1

        # Opponent lines analysis (no own chips)
        if own_count == 0:
            opp_total = opp_count + corner_count
            if opp_total == 4 and len(empty_positions) == 1:
                opp_viable_four += 1

            # Opponent threat reachability for 3+ in a row
            if opp_total >= 3 and tracker is not None and empty_positions:
                opp_three_count_for_avg += 1
                avg_prob = 0.0
                for ep in empty_positions:
                    layout_card = LAYOUT[ep.row][ep.col]
                    if layout_card is not None:
                        avg_prob += tracker.opponent_has_card_probability(layout_card)
                avg_prob /= len(empty_positions)
                opp_three_reachable_sum += avg_prob

    # Feature 36: opp_threat_reachability
    opp_three_in_a_row = float(base[6])
    if opp_three_count_for_avg > 0:
        avg_reachability = opp_three_reachable_sum / opp_three_count_for_avg
        opp_threat_reachability = opp_three_in_a_row * avg_reachability
    else:
        opp_threat_reachability = 0.0

    # Feature 37: permanent_disruption_potential
    permanent_disruption_potential = 0
    if tracker is not None:
        for r in range(10):
            for c in range(10):
                pos = Position(r, c)
                val = int(chips[r, c])
                if val not in opp_vals:
                    continue
                # Check if the card for this position is dead
                layout_card = LAYOUT[r][c]
                if layout_card is None:
                    continue
                remaining = tracker.copies_remaining_in_pool(layout_card)
                in_hand_opp = 0  # We don't know opp hand, use pool
                if remaining == 0:
                    permanent_disruption_potential += 1

    # Feature 38: hand_flexibility — unique reachable positions from hand
    reachable: set[Position] = set()
    for card in hand:
        if card.is_two_eyed_jack:
            # Can reach any empty position
            reachable |= board.empty_positions
        elif card.is_one_eyed_jack:
            pass  # Removals don't place
        else:
            for pos in CARD_TO_POSITIONS.get(card, []):
                if board.is_empty(pos):
                    reachable.add(pos)
    hand_flexibility = len(reachable)

    # Feature 39: race_advantage
    race_advantage = own_viable_four - opp_viable_four

    # Feature 40: multi_block_value
    multi_block_value = 0
    for r in range(10):
        for c in range(10):
            pos = Position(r, c)
            if int(chips[r, c]) != EMPTY or pos in CORNERS:
                continue
            opp_lines_blocked = 0
            for line_idx in POSITION_TO_LINES.get(pos, []):
                line = ALL_LINES[line_idx]
                own_in = 0
                opp_in = 0
                corner_in = 0
                for lp in line:
                    v = int(chips[lp.row, lp.col])
                    if v == team_val:
                        own_in += 1
                    elif v in opp_vals:
                        opp_in += 1
                    elif v == CORNER:
                        corner_in += 1
                if own_in == 0 and opp_in + corner_in >= 3:
                    opp_lines_blocked += 1
            if opp_lines_blocked > multi_block_value:
                multi_block_value = opp_lines_blocked

    # Feature 41: key_card_count — cards in hand that complete a 4-in-a-row
    key_card_count = 0
    for card in hand:
        if card.is_jack:
            continue
        for pos in CARD_TO_POSITIONS.get(card, []):
            if not board.is_empty(pos):
                continue
            for line_idx in POSITION_TO_LINES.get(pos, []):
                line = ALL_LINES[line_idx]
                own_count = 0
                opp_count = 0
                corner_count = 0
                for lp in line:
                    if lp == pos:
                        continue
                    v = int(chips[lp.row, lp.col])
                    if v == team_val:
                        own_count += 1
                    elif v in opp_vals:
                        opp_count += 1
                    elif v == CORNER:
                        corner_count += 1
                if opp_count == 0 and own_count + corner_count == 4:
                    key_card_count += 1
                    break  # Count card once
            else:
                continue
            break

    # Feature 42: urgency_clock — min turns until opponent could win
    urgency_clock = _compute_urgency_clock(state, team, opp_vals, tracker)

    # Feature 43: alive_line_position_score
    for pos in alive_line_positions:
        if int(chips[pos.row, pos.col]) == team_val:
            alive_line_position_score += len(POSITION_TO_LINES.get(pos, [])) / 12.0

    expert_features = np.array([
        silent_threat_count,             # 35
        corner_line_progress,            # 36
        phantom_threat_count,            # 37
        opp_threat_reachability,         # 38
        permanent_disruption_potential,  # 39
        hand_flexibility,                # 40
        race_advantage,                  # 41
        multi_block_value,               # 42
        key_card_count,                  # 43
        urgency_clock,                   # 44
        alive_line_position_score,       # 45
        open_ended_three_count,          # 46
    ], dtype=np.float64)

    return np.concatenate([base, expert_features])


def _is_fillable(pos: Position, tracker, hand: list) -> bool:
    """Check if a position can be filled (card exists somewhere)."""
    layout_card = LAYOUT[pos.row][pos.col]
    if layout_card is None:
        return False
    for c in hand:
        if c == layout_card or c.is_two_eyed_jack:
            return True
    if tracker.copies_remaining_in_pool(layout_card) > 0:
        return True
    from ...core.card import Card
    from ...core.types import Rank, Suit
    for suit in (Suit.DIAMONDS, Suit.CLUBS):
        tej = Card(Rank.JACK, suit)
        if tracker.copies_remaining_in_pool(tej) > 0:
            return True
    return False


def _compute_urgency_clock(
    state: GameState,
    team: TeamId,
    opp_vals: set[int],
    tracker: CardTracker | None,
) -> float:
    """Estimate minimum turns until opponent wins (lower = more urgent).

    Based on opponent's best lines and card availability.
    Returns a value between 1 (imminent) and 20 (far away).
    Capped for scoring stability.
    """
    board = state.board
    chips = board.chips
    sequences_to_win = state.sequences_to_win
    team_val = team.value

    opp_sequences = 0
    for t_val in opp_vals:
        opp_sequences += board.count_sequences(TeamId(t_val))

    needed_sequences = sequences_to_win - opp_sequences
    if needed_sequences <= 0:
        return 1.0  # Already lost

    # Find the best opponent lines (fewest holes to fill)
    opp_line_gaps: list[int] = []  # number of empty positions needed per viable line
    for line in ALL_LINES:
        own_count = 0
        opp_count = 0
        corner_count = 0
        empty_count = 0
        for pos in line:
            val = int(chips[pos.row, pos.col])
            if val == team_val:
                own_count += 1
            elif val in opp_vals:
                opp_count += 1
            elif val == CORNER:
                corner_count += 1
            elif val == EMPTY:
                empty_count += 1

        if own_count == 0:  # Viable for opponent
            opp_total = opp_count + corner_count
            gaps = 5 - opp_total
            if gaps > 0:
                opp_line_gaps.append(gaps)

    if not opp_line_gaps:
        return 20.0

    opp_line_gaps.sort()
    # Sum the best N gaps where N = needed_sequences
    total_turns = sum(opp_line_gaps[:needed_sequences])
    return float(max(1.0, min(20.0, total_turns)))
