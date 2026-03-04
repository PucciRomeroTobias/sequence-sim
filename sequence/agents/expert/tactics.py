"""Tier 1 (instant decisions) and Tier 2 (tactical priorities) for ExpertAgent."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...core.board import ALL_LINES, LAYOUT, POSITION_TO_LINES
from ...core.types import CORNER, CORNERS, EMPTY, Position, TeamId

if TYPE_CHECKING:
    from ...core.actions import Action
    from ...core.card_tracker import CardTracker
    from ...core.game_state import GameState


def check_instant_decisions(
    state: GameState,
    legal_actions: list[Action],
    team: TeamId,
    tracker: CardTracker | None = None,
) -> Action | None:
    """Tier 1: deterministic decisions that don't require scoring.

    Priority order:
    1. Win now — complete our N-th sequence
    2. Block win — block opponent 4-in-a-row (skip if double-threat → desperation)
    3. Winning fork — single move creates 2+ lines of 4-in-a-row
    4. Prevent opponent winning fork
    5. Force-win — create a second viable 4-in-a-row (opponent can only block one)
    """
    from ...core.actions import ActionType

    board = state.board
    chips = board.chips
    team_val = team.value

    opp_vals: set[int] = set()
    for t in range(state.num_teams):
        if t != team_val:
            opp_vals.add(t)

    # --- 1. Win Now ---
    for action in legal_actions:
        if action.action_type == ActionType.PLACE and action.position:
            next_state = state.apply_action(action)
            if next_state.board.count_sequences(team) > board.count_sequences(team):
                return action

    # --- 2. Block Win (with double-threat detection) ---
    blocking_positions: set[Position] = set()
    for line in ALL_LINES:
        opp_count = 0
        own_count = 0
        corner_count = 0
        empty_pos = None
        for pos in line:
            val = int(chips[pos.row, pos.col])
            if val in opp_vals:
                opp_count += 1
            elif val == team_val:
                own_count += 1
            elif val == CORNER:
                corner_count += 1
            elif val == EMPTY:
                empty_pos = pos

        opp_total = opp_count + corner_count
        if own_count == 0 and opp_total == 4 and empty_pos is not None:
            blocking_positions.add(empty_pos)

    if blocking_positions:
        if len(blocking_positions) >= 2:
            # Double-threat: can't block both → skip blocking, go offensive
            pass
        else:
            # Single threat: block it
            block_pos = next(iter(blocking_positions))
            jack_block: Action | None = None
            for action in legal_actions:
                if (
                    action.action_type == ActionType.PLACE
                    and action.position == block_pos
                ):
                    if action.card.is_two_eyed_jack:
                        if jack_block is None:
                            jack_block = action
                    else:
                        return action
            if jack_block is not None:
                return jack_block

    # --- 3. Winning Fork: move that creates 2+ lines of 4-in-a-row post-placement ---
    winning_fork = _find_winning_fork(state, legal_actions, team, opp_vals)
    if winning_fork is not None:
        return winning_fork

    # --- 4. Prevent Opponent Winning Fork ---
    opp_fork_block = _find_opponent_fork_block(
        state, legal_actions, team, opp_vals, tracker
    )
    if opp_fork_block is not None:
        return opp_fork_block

    # --- 5. Force-Win: we already have a viable 4-in-a-row, create a second one ---
    force_win = _find_force_win(state, legal_actions, team, opp_vals)
    if force_win is not None:
        return force_win

    return None


def check_tactical(
    state: GameState,
    legal_actions: list[Action],
    team: TeamId,
    stance: str,
    tracker: CardTracker | None = None,
) -> Action | None:
    """Tier 2: high-value tactical moves (tempo threats, building forks).

    Returns the best tactical move, or None if no clear tactical winner.
    These are scored rather than automatic — only return if clearly dominant.
    """
    from ...core.actions import ActionType

    board = state.board
    chips = board.chips
    team_val = team.value

    opp_vals: set[int] = set()
    for t in range(state.num_teams):
        if t != team_val:
            opp_vals.add(t)

    best_action: Action | None = None
    best_score = 0.0

    for action in legal_actions:
        if action.action_type != ActionType.PLACE or action.position is None:
            continue

        score = 0.0
        pos = action.position

        # Tempo threat: creates a 4-in-a-row (forces opponent to block)
        tempo = _count_four_in_a_row_after_place(pos, team_val, opp_vals, chips)
        if tempo > 0:
            score += 25.0 * tempo

        # Building fork: creates position with 2+ lines at 3-in-a-row
        building_fork = _count_three_in_a_row_lines_after_place(
            pos, team_val, opp_vals, chips
        )
        if building_fork >= 2:
            score += 15.0 * building_fork

        if score > best_score:
            best_score = score
            best_action = action

    # Only return tactical if clearly dominant (tempo threat creating 4-in-a-row)
    # Threshold 40 = tempo + some building fork synergy
    if best_score >= 40.0:
        return best_action
    return None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _find_winning_fork(
    state: GameState,
    legal_actions: list[Action],
    team: TeamId,
    opp_vals: set[int],
) -> Action | None:
    """Find a move that creates 2+ lines of 4-in-a-row simultaneously."""
    from ...core.actions import ActionType

    chips = state.board.chips
    team_val = team.value

    jack_fork: Action | None = None
    for action in legal_actions:
        if action.action_type != ActionType.PLACE or action.position is None:
            continue
        pos = action.position
        four_count = _count_four_in_a_row_after_place(pos, team_val, opp_vals, chips)
        if four_count >= 2:
            if action.card.is_two_eyed_jack:
                if jack_fork is None:
                    jack_fork = action
            else:
                return action

    return jack_fork


def _find_opponent_fork_block(
    state: GameState,
    legal_actions: list[Action],
    team: TeamId,
    opp_vals: set[int],
    tracker: CardTracker | None,
) -> Action | None:
    """Block positions where opponent could create a winning fork.

    An opponent winning fork position is an empty position where the opponent
    would get 2+ lines of 4-in-a-row after placing there. We check that the
    opponent could actually play there (card probability > 0 via tracker).
    """
    from ...core.actions import ActionType

    chips = state.board.chips
    team_val = team.value

    # Find opponent fork positions
    opp_fork_positions: set[Position] = set()
    for r in range(10):
        for c in range(10):
            pos = Position(r, c)
            if int(chips[r, c]) != EMPTY or pos in CORNERS:
                continue

            # Check how many 4-in-a-row the opponent would create
            four_count = 0
            for opp_val in opp_vals:
                four_count += _count_four_in_a_row_after_place(
                    pos, opp_val, {team_val}, chips
                )
            if four_count < 2:
                continue

            # Verify opponent can actually reach this position
            if tracker is not None:
                layout_card = LAYOUT[pos.row][pos.col]
                if layout_card is not None:
                    prob = tracker.opponent_has_card_probability(layout_card)
                    if prob < 0.05:
                        continue

            opp_fork_positions.add(pos)

    if not opp_fork_positions:
        return None

    # Try to block one of these fork positions
    from ...core.actions import ActionType as AT

    jack_block: Action | None = None
    for action in legal_actions:
        if action.action_type == AT.PLACE and action.position in opp_fork_positions:
            if action.card.is_two_eyed_jack:
                if jack_block is None:
                    jack_block = action
            else:
                return action

    return jack_block


def _find_force_win(
    state: GameState,
    legal_actions: list[Action],
    team: TeamId,
    opp_vals: set[int],
) -> Action | None:
    """If we already have a viable 4-in-a-row, find a move creating a second one.

    The opponent can only block one, so creating two forces a win next turn.
    """
    from ...core.actions import ActionType

    chips = state.board.chips
    team_val = team.value

    # Count existing viable 4-in-a-rows
    existing_viable_four = 0
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
        if opp_count == 0 and own_count + corner_count == 4 and empty_count == 1:
            existing_viable_four += 1

    if existing_viable_four < 1:
        return None

    # Find a move that creates an additional viable 4-in-a-row on a DIFFERENT line
    jack_option: Action | None = None
    for action in legal_actions:
        if action.action_type != ActionType.PLACE or action.position is None:
            continue
        pos = action.position
        new_viable = _count_four_in_a_row_after_place(pos, team_val, opp_vals, chips)
        if new_viable >= 1:
            if action.card.is_two_eyed_jack:
                if jack_option is None:
                    jack_option = action
            else:
                return action

    return jack_option


def _count_four_in_a_row_after_place(
    pos: Position,
    team_val: int,
    opp_vals: set[int],
    chips,
) -> int:
    """Count how many lines become 4-in-a-row (with 1 empty hole) after placing at pos.

    Simulates placing a chip at pos for team_val, then counts lines through pos
    that have exactly own+corner=4 and 1 empty (the remaining hole).
    Note: the placed chip itself makes own_total go from 3 to 4, plus the
    hole position is a different empty cell.
    """
    line_indices = POSITION_TO_LINES.get(pos, [])
    count = 0

    for line_idx in line_indices:
        line = ALL_LINES[line_idx]
        own_count = 0
        opp_count = 0
        corner_count = 0
        for lp in line:
            if lp == pos:
                # This is where we're placing — count as own
                own_count += 1
                continue
            val = int(chips[lp.row, lp.col])
            if val == team_val:
                own_count += 1
            elif val in opp_vals:
                opp_count += 1
            elif val == CORNER:
                corner_count += 1

        own_total = own_count + corner_count
        if opp_count == 0 and own_total == 4:
            count += 1

    return count


def _count_three_in_a_row_lines_after_place(
    pos: Position,
    team_val: int,
    opp_vals: set[int],
    chips,
) -> int:
    """Count how many lines become 3-in-a-row (no opponent) after placing at pos."""
    line_indices = POSITION_TO_LINES.get(pos, [])
    count = 0

    for line_idx in line_indices:
        line = ALL_LINES[line_idx]
        own_count = 0
        opp_count = 0
        corner_count = 0
        for lp in line:
            if lp == pos:
                own_count += 1
                continue
            val = int(chips[lp.row, lp.col])
            if val == team_val:
                own_count += 1
            elif val in opp_vals:
                opp_count += 1
            elif val == CORNER:
                corner_count += 1

        own_total = own_count + corner_count
        if opp_count == 0 and own_total == 3:
            count += 1

    return count
