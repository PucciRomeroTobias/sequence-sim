"""Sequence game board with chip placement and sequence detection."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from .card import Card
from .types import CORNER, CORNERS, EMPTY, Position, TeamId

# Official Sequence board layout (10x10).
# "Free" corners are represented as None.
_LAYOUT_STRINGS: list[list[str | None]] = [
    [None, "2S", "3S", "4S", "5S", "6S", "7S", "8S", "9S", None],
    ["6C", "5C", "4C", "3C", "2C", "AH", "KH", "QH", "10H", "10S"],
    ["7C", "AS", "2D", "3D", "4D", "5D", "6D", "7D", "9H", "QS"],
    ["8C", "KS", "6C", "5C", "4C", "3C", "2C", "8D", "8H", "KS"],
    ["9C", "QS", "7C", "6H", "5H", "4H", "AH", "9D", "7H", "AS"],
    ["10C", "10S", "8C", "7H", "2H", "3H", "KH", "10D", "6H", "2D"],
    ["QC", "9S", "9C", "8H", "9H", "10H", "QH", "QD", "5H", "3D"],
    ["KC", "8S", "10C", "QC", "KC", "AC", "AD", "KD", "4H", "4D"],
    ["AC", "7S", "6S", "5S", "4S", "3S", "2S", "2H", "3H", "5D"],
    [None, "AD", "KD", "QD", "10D", "9D", "8D", "7D", "6D", None],
]

# Pre-compute the board layout as Card objects
LAYOUT: list[list[Card | None]] = []
for row in _LAYOUT_STRINGS:
    card_row: list[Card | None] = []
    for cell in row:
        card_row.append(Card.from_str(cell) if cell is not None else None)
    LAYOUT.append(card_row)

# Pre-compute card -> board positions mapping.
# Each non-Jack card appears exactly 2 times on the board (2 positions).
# Jacks do not appear on the board (they are special action cards).
CARD_TO_POSITIONS: dict[Card, list[Position]] = defaultdict(list)
for r, row in enumerate(LAYOUT):
    for c, card in enumerate(row):
        if card is not None and not card.is_jack:
            CARD_TO_POSITIONS[card].append(Position(r, c))
# Freeze it
CARD_TO_POSITIONS = dict(CARD_TO_POSITIONS)

# Pre-compute all possible lines of 5 on the board (for sequence detection).
# Each line is a tuple of 5 Positions.
ALL_LINES: list[tuple[Position, ...]] = []
for r in range(10):
    for c in range(10):
        # Horizontal
        if c + 4 < 10:
            ALL_LINES.append(tuple(Position(r, c + i) for i in range(5)))
        # Vertical
        if r + 4 < 10:
            ALL_LINES.append(tuple(Position(r + i, c) for i in range(5)))
        # Diagonal down-right
        if r + 4 < 10 and c + 4 < 10:
            ALL_LINES.append(tuple(Position(r + i, c + i) for i in range(5)))
        # Diagonal down-left
        if r + 4 < 10 and c - 4 >= 0:
            ALL_LINES.append(tuple(Position(r + i, c - i) for i in range(5)))

# Pre-compute which lines pass through each position
POSITION_TO_LINES: dict[Position, list[int]] = defaultdict(list)
for idx, line in enumerate(ALL_LINES):
    for pos in line:
        POSITION_TO_LINES[pos].append(idx)
POSITION_TO_LINES = dict(POSITION_TO_LINES)


class Board:
    """10x10 Sequence game board tracking chip placement and sequences."""

    __slots__ = ("_chips", "_empty_positions", "_sequences")

    def __init__(self) -> None:
        # -1 = empty, 0/1/2 = team chip, 3 = corner (wild)
        self._chips: np.ndarray = np.full((10, 10), EMPTY, dtype=np.int8)
        # Mark corners
        for pos in CORNERS:
            self._chips[pos.row, pos.col] = CORNER
        # Track empty positions (excludes corners)
        self._empty_positions: set[Position] = set()
        for r in range(10):
            for c in range(10):
                if self._chips[r, c] == EMPTY:
                    self._empty_positions.add(Position(r, c))
        # Track completed sequences: set of frozensets of positions
        self._sequences: dict[int, set[frozenset[Position]]] = {
            t.value: set() for t in TeamId
        }

    @property
    def chips(self) -> np.ndarray:
        return self._chips

    @property
    def empty_positions(self) -> set[Position]:
        return self._empty_positions

    def place_chip(self, pos: Position, team: TeamId) -> None:
        if self._chips[pos.row, pos.col] != EMPTY:
            raise ValueError(f"Position {pos} is not empty")
        self._chips[pos.row, pos.col] = team.value
        self._empty_positions.discard(pos)

    def remove_chip(self, pos: Position) -> None:
        val = self._chips[pos.row, pos.col]
        if val == EMPTY or val == CORNER:
            raise ValueError(f"Cannot remove chip from {pos} (value={val})")
        self._chips[pos.row, pos.col] = EMPTY
        self._empty_positions.add(pos)

    def get_chip(self, pos: Position) -> int:
        return int(self._chips[pos.row, pos.col])

    def is_empty(self, pos: Position) -> bool:
        return self._chips[pos.row, pos.col] == EMPTY

    def is_corner(self, pos: Position) -> bool:
        return pos in CORNERS

    def check_new_sequences(self, pos: Position, team: TeamId) -> list[frozenset[Position]]:
        """Check if placing at pos created any new sequences for team.

        Returns list of newly completed sequences (each a frozenset of 5 positions).
        """
        new_seqs: list[frozenset[Position]] = []
        team_val = team.value
        line_indices = POSITION_TO_LINES.get(pos, [])

        for line_idx in line_indices:
            line = ALL_LINES[line_idx]
            line_frozen = frozenset(line)
            # Skip if this line is already a completed sequence for this team
            if line_frozen in self._sequences[team_val]:
                continue
            # Check if all 5 positions have team's chip or corner
            if all(
                self._chips[p.row, p.col] == team_val
                or self._chips[p.row, p.col] == CORNER
                for p in line
            ):
                new_seqs.append(line_frozen)
                self._sequences[team_val].add(line_frozen)

        return new_seqs

    def is_part_of_completed_sequence(self, pos: Position, team: TeamId) -> bool:
        """Check if a position is part of a completed sequence for any team except the given one."""
        for t_val, seqs in self._sequences.items():
            if t_val == team.value:
                continue
            for seq in seqs:
                if pos in seq:
                    return True
        return False

    def is_part_of_own_sequence(self, pos: Position, team: TeamId) -> bool:
        """Check if a position is part of a completed sequence for the given team."""
        for seq in self._sequences[team.value]:
            if pos in seq:
                return True
        return False

    def count_sequences(self, team: TeamId) -> int:
        return len(self._sequences[team.value])

    def get_all_sequences(self, team: TeamId) -> set[frozenset[Position]]:
        return self._sequences[team.value]

    def copy(self) -> Board:
        new = Board.__new__(Board)
        new._chips = self._chips.copy()
        new._empty_positions = set(self._empty_positions)
        new._sequences = {
            t: set(seqs) for t, seqs in self._sequences.items()
        }
        return new

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Board):
            return NotImplemented
        return np.array_equal(self._chips, other._chips)

    def to_list(self) -> list[list[int]]:
        """Serialize board state as nested list for JSON."""
        return self._chips.tolist()

    @classmethod
    def from_list(cls, data: list[list[int]]) -> Board:
        """Deserialize board state from nested list."""
        board = cls.__new__(cls)
        board._chips = np.array(data, dtype=np.int8)
        board._empty_positions = set()
        for r in range(10):
            for c in range(10):
                if board._chips[r, c] == EMPTY:
                    board._empty_positions.add(Position(r, c))
        board._sequences = {t.value: set() for t in TeamId}
        # Rebuild sequence tracking by scanning all lines
        for team in TeamId:
            team_val = team.value
            for line in ALL_LINES:
                if all(
                    board._chips[p.row, p.col] == team_val
                    or board._chips[p.row, p.col] == CORNER
                    for p in line
                ):
                    board._sequences[team_val].add(frozenset(line))
        return board
