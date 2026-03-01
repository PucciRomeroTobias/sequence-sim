"""Tests for Board class."""

import numpy as np

from sequence.core.board import ALL_LINES, CARD_TO_POSITIONS, LAYOUT, Board
from sequence.core.card import Card
from sequence.core.types import CORNERS, EMPTY, Position, Rank, Suit, TeamId


def test_layout_dimensions():
    assert len(LAYOUT) == 10
    for row in LAYOUT:
        assert len(row) == 10


def test_corners_are_none():
    for pos in CORNERS:
        assert LAYOUT[pos.row][pos.col] is None


def test_card_to_positions_has_two_entries():
    """Each non-Jack card should appear exactly 2 times on the board."""
    for card, positions in CARD_TO_POSITIONS.items():
        assert len(positions) == 2, f"{card} has {len(positions)} positions"
        assert not card.is_jack


def test_jacks_not_in_positions():
    """Jacks should not appear in the card-to-positions mapping."""
    for suit in Suit:
        jack = Card(Rank.JACK, suit)
        assert jack not in CARD_TO_POSITIONS


def test_board_initial_state():
    board = Board()
    # Corners should be marked as 3
    for pos in CORNERS:
        assert board.get_chip(pos) == 3
    # All other cells should be empty
    for r in range(10):
        for c in range(10):
            pos = Position(r, c)
            if pos not in CORNERS:
                assert board.is_empty(pos)


def test_place_and_remove_chip():
    board = Board()
    pos = Position(1, 1)
    board.place_chip(pos, TeamId.TEAM_0)
    assert board.get_chip(pos) == 0
    assert not board.is_empty(pos)
    assert pos not in board.empty_positions

    board.remove_chip(pos)
    assert board.is_empty(pos)
    assert pos in board.empty_positions


def test_horizontal_sequence():
    board = Board()
    team = TeamId.TEAM_0
    # Place 5 in a row horizontally at row 5, cols 0-4
    positions = [Position(5, c) for c in range(5)]
    for pos in positions:
        board.place_chip(pos, team)
    # Check sequence from last placed
    seqs = board.check_new_sequences(positions[-1], team)
    assert len(seqs) >= 1


def test_vertical_sequence():
    board = Board()
    team = TeamId.TEAM_1
    positions = [Position(r, 5) for r in range(5)]
    for pos in positions:
        board.place_chip(pos, team)
    seqs = board.check_new_sequences(positions[-1], team)
    assert len(seqs) >= 1


def test_diagonal_sequence():
    board = Board()
    team = TeamId.TEAM_0
    positions = [Position(i, i) for i in range(1, 6)]
    for pos in positions:
        board.place_chip(pos, team)
    seqs = board.check_new_sequences(positions[-1], team)
    assert len(seqs) >= 1


def test_corner_counts_in_sequence():
    """Corners (value 3) should count as wild for any team's sequence."""
    board = Board()
    team = TeamId.TEAM_0
    # Row 0: corner at (0,0), place at (0,1), (0,2), (0,3), (0,4)
    for c in range(1, 5):
        board.place_chip(Position(0, c), team)
    seqs = board.check_new_sequences(Position(0, 4), team)
    # Should form a sequence: corner + 4 chips = 5
    assert len(seqs) >= 1


def test_no_sequence_with_4():
    board = Board()
    team = TeamId.TEAM_0
    for c in range(4):
        board.place_chip(Position(3, c), team)
    seqs = board.check_new_sequences(Position(3, 3), team)
    assert len(seqs) == 0


def test_count_sequences():
    board = Board()
    team = TeamId.TEAM_0
    assert board.count_sequences(team) == 0
    # Create a sequence
    for c in range(5):
        board.place_chip(Position(2, c), team)
    board.check_new_sequences(Position(2, 4), team)
    assert board.count_sequences(team) >= 1


def test_board_copy_independence():
    board = Board()
    board.place_chip(Position(1, 1), TeamId.TEAM_0)
    copy = board.copy()
    copy.place_chip(Position(2, 2), TeamId.TEAM_1)
    # Original should not be affected
    assert board.is_empty(Position(2, 2))
    assert not copy.is_empty(Position(2, 2))


def test_is_part_of_completed_sequence():
    board = Board()
    team = TeamId.TEAM_0
    for c in range(5):
        board.place_chip(Position(4, c), team)
    board.check_new_sequences(Position(4, 4), team)

    # These positions should be part of team 0's sequence
    for c in range(5):
        assert board.is_part_of_own_sequence(Position(4, c), team)


def test_all_lines_count():
    """Verify the number of possible 5-in-a-row lines on a 10x10 board."""
    # Horizontal: 10 rows * 6 starting cols = 60
    # Vertical: 6 starting rows * 10 cols = 60
    # Diag down-right: 6 * 6 = 36
    # Diag down-left: 6 * 6 = 36
    assert len(ALL_LINES) == 60 + 60 + 36 + 36


def test_board_serialization():
    board = Board()
    board.place_chip(Position(3, 3), TeamId.TEAM_0)
    board.place_chip(Position(5, 5), TeamId.TEAM_1)

    data = board.to_list()
    restored = Board.from_list(data)
    assert np.array_equal(board.chips, restored.chips)
