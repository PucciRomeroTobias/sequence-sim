"""Stance detection and dynamic weight multipliers for the ExpertAgent."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...core.board import ALL_LINES
from ...core.types import CORNER, EMPTY, Position, TeamId

if TYPE_CHECKING:
    from ...core.card_tracker import CardTracker
    from ...core.game_state import GameState


# Stance identifiers
OPENING = "opening"
BUILDING = "building"
CLOSE_OUT = "close_out"
DESPERATE_DEFENSE = "desperate_defense"
RACE = "race"
DESPERATION_OFFENSE = "desperation_offense"

# Weight multipliers per stance: (offense_mult, defense_mult, position_mult)
# Conservative multipliers to avoid distorting genetically-optimized base weights
STANCE_MULTIPLIERS: dict[str, tuple[float, float, float]] = {
    OPENING: (1.15, 0.85, 1.2),
    BUILDING: (1.0, 1.0, 1.0),
    CLOSE_OUT: (1.3, 1.0, 1.0),
    DESPERATE_DEFENSE: (1.0, 1.5, 1.0),
    RACE: (1.5, 0.7, 0.9),
    DESPERATION_OFFENSE: (2.0, 0.3, 0.7),
}

# Indices classified as offensive, defensive, or positional for multiplier application.
# Offensive: completed_sequences(0), four(1), three(2), two(3), viable_four(18),
#            viable_three(19), fork_count(22), sequence_proximity(27),
#            anchor_overlap(29), card_monopoly(26)
OFFENSIVE_INDICES = {0, 1, 2, 3, 18, 19, 22, 26, 27, 29}
# Defensive: opp_completed(4), opp_four(5), opp_three(6), opp_blockable(20),
#            opp_unblockable(21), opp_dead_lines(25)
DEFENSIVE_INDICES = {4, 5, 6, 20, 21, 25}
# Positional: center(9), corner_adj(10), shared_line(15), position_line_score(28),
#             chip_clustering(30)
POSITIONAL_INDICES = {9, 10, 15, 28, 30}


def compute_stance(
    state: GameState,
    team: TeamId,
    tracker: CardTracker | None = None,
) -> str:
    """Determine the current game stance for dynamic weight selection.

    Priority order (first match wins):
    1. desperation_offense — opponent has 2+ unblockable 4-in-a-rows
    2. race — both sides have viable 4-in-a-row
    3. desperate_defense — opponent has completed >= 1 sequence
    4. close_out — we have completed >= 1 sequence
    5. opening — turn <= 10
    6. building — default
    """
    board = state.board
    chips = board.chips
    team_val = team.value

    opp_vals: set[int] = set()
    for t in range(state.num_teams):
        if t != team_val:
            opp_vals.add(t)

    own_sequences = board.count_sequences(team)
    opp_sequences = 0
    for t in range(state.num_teams):
        if t != team_val:
            opp_sequences += board.count_sequences(TeamId(t))

    # Scan lines for viable 4-in-a-rows and unblockable threats
    own_viable_four = 0
    opp_viable_four = 0
    opp_unblockable_four = 0
    hand = state.hands.get(team_val, [])

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

        if opp_count == 0 and own_total == 4 and len(empty_positions) == 1:
            own_viable_four += 1

        if own_count == 0 and opp_total == 4 and len(empty_positions) == 1:
            opp_viable_four += 1
            # Check if we can block
            can_block = _can_place_at(empty_positions[0], team_val, hand, chips)
            if not can_block:
                opp_unblockable_four += 1

    # Priority-based stance selection
    if opp_unblockable_four >= 2:
        return DESPERATION_OFFENSE
    if own_viable_four >= 1 and opp_viable_four >= 1:
        return RACE
    if opp_sequences >= 1:
        return DESPERATE_DEFENSE
    if own_sequences >= 1:
        return CLOSE_OUT
    if state.turn_number <= 10:
        return OPENING
    return BUILDING


def get_weight_multipliers(stance: str) -> tuple[float, float, float]:
    """Return (offense_mult, defense_mult, position_mult) for a stance."""
    return STANCE_MULTIPLIERS.get(stance, (1.0, 1.0, 1.0))


def _can_place_at(
    pos: Position, team_val: int, hand: list, chips,
) -> bool:
    """Check if we can currently place a chip at this position from our hand."""
    from ...core.board import LAYOUT

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
