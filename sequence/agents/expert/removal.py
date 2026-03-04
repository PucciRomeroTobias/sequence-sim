"""Intelligent removal scoring for one-eyed jack targeting."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...core.board import ALL_LINES, LAYOUT, POSITION_TO_LINES
from ...core.types import CORNER, EMPTY, Position, TeamId

if TYPE_CHECKING:
    from ...core.actions import Action
    from ...core.card_tracker import CardTracker
    from ...core.game_state import GameState


def score_removal(
    action: Action,
    state: GameState,
    team: TeamId,
    tracker: CardTracker | None = None,
) -> float:
    """Score a REMOVE action based on multiple criteria.

    Criteria:
    - Breaks opponent 4-in-a-row (+100)
    - Breaks opponent 3-in-a-row (+40)
    - Removal is permanent: card at that position is dead (+30)
    - Unblocks our line: removed chip was only obstacle in a line with 3+ own chips (+50)
    - Multi-disruption: removed chip participates in 2+ opponent lines (+15 each extra)
    """
    assert action.position is not None
    pos = action.position
    board = state.board
    chips = board.chips
    team_val = team.value

    opp_vals: set[int] = set()
    for t in range(state.num_teams):
        if t != team_val:
            opp_vals.add(t)

    score = 0.0
    opp_lines_disrupted = 0
    line_indices = POSITION_TO_LINES.get(pos, [])

    for line_idx in line_indices:
        line = ALL_LINES[line_idx]
        own_count = 0
        opp_count = 0
        corner_count = 0
        for lp in line:
            val = int(chips[lp.row, lp.col])
            if val == team_val:
                own_count += 1
            elif val in opp_vals:
                opp_count += 1
            elif val == CORNER:
                corner_count += 1

        opp_total = opp_count + corner_count

        # Target chip is one of the opp_count — after removal, opp loses 1
        if own_count == 0:
            if opp_total == 5:
                # Breaking a completed sequence isn't possible (protected), skip
                pass
            elif opp_total == 4 and opp_count >= 1:
                # Breaking opponent 4-in-a-row
                score += 100.0
                opp_lines_disrupted += 1
            elif opp_total == 3 and opp_count >= 1:
                # Breaking opponent 3-in-a-row
                score += 40.0
                opp_lines_disrupted += 1
            elif opp_count >= 1:
                opp_lines_disrupted += 1

        # Unblocks our line: this was the ONLY opponent chip in a line with 3+ own
        if opp_count == 1 and own_count + corner_count >= 3:
            score += 50.0

    # Multi-disruption bonus
    if opp_lines_disrupted > 1:
        score += 15.0 * (opp_lines_disrupted - 1)

    # Permanent removal: check if the card at this position is dead
    if tracker is not None:
        layout_card = LAYOUT[pos.row][pos.col]
        if layout_card is not None:
            # After removal the position is empty. Check if the card has copies left.
            # If no copies remain (played + discarded + in hand = total), removal is permanent.
            remaining = tracker.copies_remaining_in_pool(layout_card)
            in_hand = tracker.copies_in_hand(layout_card)
            if remaining == 0 and in_hand == 0:
                score += 30.0

    return score
