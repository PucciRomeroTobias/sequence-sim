"""Card representation for the Sequence game."""

from __future__ import annotations

from dataclasses import dataclass

from .types import Rank, Suit


@dataclass(frozen=True, slots=True)
class Card:
    rank: Rank
    suit: Suit

    @property
    def is_one_eyed_jack(self) -> bool:
        """One-eyed Jacks (JH, JS) remove an opponent's chip."""
        return self.rank == Rank.JACK and self.suit in (Suit.HEARTS, Suit.SPADES)

    @property
    def is_two_eyed_jack(self) -> bool:
        """Two-eyed Jacks (JD, JC) are wild — place on any empty space."""
        return self.rank == Rank.JACK and self.suit in (Suit.DIAMONDS, Suit.CLUBS)

    @property
    def is_jack(self) -> bool:
        return self.rank == Rank.JACK

    @classmethod
    def from_str(cls, s: str) -> Card:
        """Parse a card string like '10D', 'JS', '2H'."""
        s = s.strip()
        if len(s) < 2:
            raise ValueError(f"Invalid card string: {s!r}")
        if s[:-1] == "10":
            rank_str, suit_str = "10", s[-1]
        else:
            rank_str, suit_str = s[:-1], s[-1]
        rank = Rank(rank_str)
        suit = Suit(suit_str)
        return cls(rank=rank, suit=suit)

    def __str__(self) -> str:
        return f"{self.rank.value}{self.suit.value}"

    def __repr__(self) -> str:
        return f"Card({self!s})"


def make_full_deck() -> list[Card]:
    """Create a list of all 52 unique cards (one standard deck)."""
    return [Card(rank=r, suit=s) for s in Suit for r in Rank]
