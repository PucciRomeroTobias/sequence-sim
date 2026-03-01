"""Base types for the Sequence game engine."""

from __future__ import annotations

from enum import Enum, IntEnum
from typing import NamedTuple


class Suit(Enum):
    SPADES = "S"
    HEARTS = "H"
    DIAMONDS = "D"
    CLUBS = "C"

    def __repr__(self) -> str:
        return f"Suit.{self.name}"


class Rank(Enum):
    TWO = "2"
    THREE = "3"
    FOUR = "4"
    FIVE = "5"
    SIX = "6"
    SEVEN = "7"
    EIGHT = "8"
    NINE = "9"
    TEN = "10"
    JACK = "J"
    QUEEN = "Q"
    KING = "K"
    ACE = "A"

    def __repr__(self) -> str:
        return f"Rank.{self.name}"


class TeamId(IntEnum):
    TEAM_0 = 0
    TEAM_1 = 1
    TEAM_2 = 2


class Position(NamedTuple):
    row: int
    col: int


# The four corner positions are free (wild) for all teams
CORNERS: frozenset[Position] = frozenset(
    [Position(0, 0), Position(0, 9), Position(9, 0), Position(9, 9)]
)

# Chip values on the board
EMPTY = -1
CORNER = 3
