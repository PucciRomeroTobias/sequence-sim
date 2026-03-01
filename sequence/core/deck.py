"""Deck management for the Sequence game (2 standard decks = 104 cards)."""

from __future__ import annotations

import random

from .card import Card, make_full_deck


class Deck:
    """A deck of 104 cards (2 standard decks) with draw, discard, and reshuffle."""

    __slots__ = ("_cards", "_discard_pile", "_rng")

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)
        # 2 complete decks
        self._cards: list[Card] = make_full_deck() + make_full_deck()
        self._rng.shuffle(self._cards)
        self._discard_pile: list[Card] = []

    @property
    def remaining(self) -> int:
        return len(self._cards)

    @property
    def discard_count(self) -> int:
        return len(self._discard_pile)

    def draw(self) -> Card | None:
        """Draw a card. Reshuffles discard pile if draw pile is empty.
        Returns None if both piles are empty.
        """
        if not self._cards:
            if not self._discard_pile:
                return None
            self._reshuffle()
        return self._cards.pop()

    def discard(self, card: Card) -> None:
        self._discard_pile.append(card)

    def _reshuffle(self) -> None:
        self._cards = self._discard_pile
        self._discard_pile = []
        self._rng.shuffle(self._cards)

    def peek_remaining(self) -> list[Card]:
        """Return a copy of remaining cards (for determinization in MCTS)."""
        return list(self._cards)

    def peek_discarded(self) -> list[Card]:
        """Return a copy of discarded cards."""
        return list(self._discard_pile)

    def copy(self, seed: int | None = None) -> Deck:
        """Create a copy of this deck with a new RNG seed."""
        new = Deck.__new__(Deck)
        new._cards = list(self._cards)
        new._discard_pile = list(self._discard_pile)
        new._rng = random.Random(seed) if seed is not None else random.Random()
        # Copy RNG state from original if no new seed
        if seed is None:
            new._rng.setstate(self._rng.getstate())
        return new
