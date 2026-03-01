"""Tests for Deck class."""

from sequence.core.deck import Deck


def test_deck_has_104_cards():
    deck = Deck(seed=42)
    assert deck.remaining == 104


def test_draw_reduces_count():
    deck = Deck(seed=42)
    card = deck.draw()
    assert card is not None
    assert deck.remaining == 103


def test_draw_all_cards():
    deck = Deck(seed=42)
    cards = []
    for _ in range(104):
        c = deck.draw()
        assert c is not None
        cards.append(c)
    assert deck.remaining == 0
    assert len(cards) == 104


def test_reshuffle_on_empty():
    deck = Deck(seed=42)
    # Draw all and discard all
    drawn = []
    for _ in range(104):
        c = deck.draw()
        assert c is not None
        drawn.append(c)
    assert deck.remaining == 0

    for c in drawn:
        deck.discard(c)
    assert deck.discard_count == 104

    # Drawing should trigger reshuffle
    card = deck.draw()
    assert card is not None
    assert deck.remaining == 103
    assert deck.discard_count == 0


def test_draw_returns_none_when_fully_empty():
    deck = Deck(seed=42)
    for _ in range(104):
        deck.draw()
    # No cards drawn, no discards
    assert deck.draw() is None


def test_deterministic_with_seed():
    deck1 = Deck(seed=123)
    deck2 = Deck(seed=123)
    for _ in range(50):
        assert deck1.draw() == deck2.draw()


def test_copy_independence():
    deck = Deck(seed=42)
    for _ in range(10):
        deck.draw()
    copy = deck.copy(seed=99)
    # Drawing from copy shouldn't affect original
    copy.draw()
    assert deck.remaining == 94
    assert copy.remaining == 93
