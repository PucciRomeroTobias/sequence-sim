"""Tests for Card class."""

from sequence.core.card import Card, make_full_deck
from sequence.core.types import Rank, Suit


def test_card_creation():
    c = Card(Rank.ACE, Suit.SPADES)
    assert c.rank == Rank.ACE
    assert c.suit == Suit.SPADES


def test_card_from_str():
    assert Card.from_str("10D") == Card(Rank.TEN, Suit.DIAMONDS)
    assert Card.from_str("JS") == Card(Rank.JACK, Suit.SPADES)
    assert Card.from_str("2H") == Card(Rank.TWO, Suit.HEARTS)
    assert Card.from_str("AC") == Card(Rank.ACE, Suit.CLUBS)
    assert Card.from_str("KS") == Card(Rank.KING, Suit.SPADES)


def test_card_str():
    assert str(Card(Rank.TEN, Suit.DIAMONDS)) == "10D"
    assert str(Card(Rank.JACK, Suit.SPADES)) == "JS"
    assert str(Card(Rank.TWO, Suit.HEARTS)) == "2H"


def test_card_roundtrip():
    for card in make_full_deck():
        assert Card.from_str(str(card)) == card


def test_one_eyed_jack():
    assert Card(Rank.JACK, Suit.HEARTS).is_one_eyed_jack
    assert Card(Rank.JACK, Suit.SPADES).is_one_eyed_jack
    assert not Card(Rank.JACK, Suit.DIAMONDS).is_one_eyed_jack
    assert not Card(Rank.JACK, Suit.CLUBS).is_one_eyed_jack


def test_two_eyed_jack():
    assert Card(Rank.JACK, Suit.DIAMONDS).is_two_eyed_jack
    assert Card(Rank.JACK, Suit.CLUBS).is_two_eyed_jack
    assert not Card(Rank.JACK, Suit.HEARTS).is_two_eyed_jack
    assert not Card(Rank.JACK, Suit.SPADES).is_two_eyed_jack


def test_card_hashable():
    c1 = Card(Rank.ACE, Suit.SPADES)
    c2 = Card(Rank.ACE, Suit.SPADES)
    assert c1 == c2
    assert hash(c1) == hash(c2)
    s = {c1, c2}
    assert len(s) == 1


def test_full_deck():
    deck = make_full_deck()
    assert len(deck) == 52
    assert len(set(deck)) == 52
