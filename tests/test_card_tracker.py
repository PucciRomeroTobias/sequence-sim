"""Tests for CardTracker — card counting and public information tracking."""

from __future__ import annotations

import pytest

from sequence.core.actions import Action, ActionType
from sequence.core.board import Board, CARD_TO_POSITIONS
from sequence.core.card import Card, make_full_deck
from sequence.core.card_tracker import CardTracker
from sequence.core.deck import Deck
from sequence.core.game_state import GameState
from sequence.core.types import EMPTY, Position, Rank, Suit, TeamId


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tracker(team: TeamId = TeamId.TEAM_0) -> CardTracker:
    return CardTracker(team, num_teams=2)


def _card(s: str) -> Card:
    return Card.from_str(s)


def _make_state(seed: int = 42) -> GameState:
    board = Board()
    deck = Deck(seed=seed)
    hands: dict[int, list[Card]] = {}
    for t in range(2):
        hand: list[Card] = []
        for _ in range(7):
            c = deck.draw()
            if c is not None:
                hand.append(c)
        hands[t] = hand
    return GameState(
        board=board, hands=hands, deck=deck,
        current_team=TeamId.TEAM_0, num_teams=2, sequences_to_win=2,
    )


# ---------------------------------------------------------------------------
# Basic tracking tests
# ---------------------------------------------------------------------------


class TestCardTrackerBasics:
    def test_initial_state(self):
        tracker = _make_tracker()
        card = _card("2S")
        assert tracker.copies_played(card) == 0
        assert tracker.copies_discarded(card) == 0
        assert tracker.copies_used(card) == 0

    def test_track_place_action(self):
        tracker = _make_tracker()
        card = _card("2S")
        pos = CARD_TO_POSITIONS[card][0]
        action = Action(card, pos, ActionType.PLACE)
        tracker.on_action(action, TeamId.TEAM_0)

        assert tracker.copies_played(card) == 1
        assert tracker.copies_used(card) == 1

    def test_track_dead_card_discard(self):
        tracker = _make_tracker()
        card = _card("3H")
        action = Action(card, None, ActionType.DEAD_CARD_DISCARD)
        tracker.on_action(action, TeamId.TEAM_1)

        assert tracker.copies_discarded(card) == 1
        assert tracker.copies_played(card) == 0
        assert tracker.copies_used(card) == 1

    def test_track_remove_action(self):
        """One-eyed jack removal consumes the jack card."""
        tracker = _make_tracker()
        jack = _card("JH")  # one-eyed jack
        pos = Position(3, 3)
        action = Action(jack, pos, ActionType.REMOVE)
        tracker.on_action(action, TeamId.TEAM_0)

        assert tracker.copies_played(jack) == 1

    def test_multiple_copies_tracked(self):
        tracker = _make_tracker()
        card = _card("5D")
        pos = CARD_TO_POSITIONS[card][0]
        action = Action(card, pos, ActionType.PLACE)

        tracker.on_action(action, TeamId.TEAM_0)
        tracker.on_action(action, TeamId.TEAM_1)

        assert tracker.copies_played(card) == 2
        assert tracker.copies_used(card) == 2


# ---------------------------------------------------------------------------
# Pool / remaining tests
# ---------------------------------------------------------------------------


class TestRemainingPool:
    def test_initial_remaining(self):
        tracker = _make_tracker()
        card = _card("2S")
        # No hand synced, no actions → full 2 copies remain
        assert tracker.copies_remaining_in_pool(card) == 2

    def test_remaining_after_play(self):
        tracker = _make_tracker()
        card = _card("2S")
        pos = CARD_TO_POSITIONS[card][0]
        action = Action(card, pos, ActionType.PLACE)
        tracker.on_action(action, TeamId.TEAM_0)

        assert tracker.copies_remaining_in_pool(card) == 1

    def test_remaining_accounts_for_hand(self):
        tracker = _make_tracker()
        card = _card("4H")
        tracker.sync_hand([card, _card("5S")])

        # 2 total - 0 used - 1 in hand = 1
        assert tracker.copies_remaining_in_pool(card) == 1

    def test_remaining_never_negative(self):
        tracker = _make_tracker()
        card = _card("2S")
        # Simulate both copies played
        pos = CARD_TO_POSITIONS[card][0]
        action = Action(card, pos, ActionType.PLACE)
        tracker.on_action(action, TeamId.TEAM_0)
        tracker.on_action(action, TeamId.TEAM_1)

        assert tracker.copies_remaining_in_pool(card) == 0

    def test_jack_in_pool(self):
        """Jacks have 2 copies per suit in 2 decks (4 jacks * 2 = 8 total jacks)."""
        tracker = _make_tracker()
        jd = _card("JD")
        # Each unique jack has 2 copies in 2 decks
        assert tracker.copies_remaining_in_pool(jd) == 2


# ---------------------------------------------------------------------------
# Unknown pool for MCTS
# ---------------------------------------------------------------------------


class TestUnknownPool:
    def test_initial_pool_size(self):
        tracker = _make_tracker()
        pool = tracker.get_unknown_card_pool()
        # 104 total cards in 2 decks, no hand synced
        assert len(pool) == 104

    def test_pool_excludes_hand(self):
        tracker = _make_tracker()
        hand = [_card("2S"), _card("3H"), _card("4D")]
        tracker.sync_hand(hand)
        pool = tracker.get_unknown_card_pool()
        assert len(pool) == 104 - 3

    def test_pool_excludes_played(self):
        tracker = _make_tracker()
        card = _card("5S")
        pos = CARD_TO_POSITIONS[card][0]
        action = Action(card, pos, ActionType.PLACE)
        tracker.on_action(action, TeamId.TEAM_0)

        pool = tracker.get_unknown_card_pool()
        assert len(pool) == 103

    def test_pool_excludes_discarded(self):
        tracker = _make_tracker()
        card = _card("6C")
        action = Action(card, None, ActionType.DEAD_CARD_DISCARD)
        tracker.on_action(action, TeamId.TEAM_0)

        pool = tracker.get_unknown_card_pool()
        assert len(pool) == 103

    def test_pool_combined(self):
        tracker = _make_tracker()
        # Play one card
        c1 = _card("2S")
        tracker.on_action(
            Action(c1, CARD_TO_POSITIONS[c1][0], ActionType.PLACE), TeamId.TEAM_0
        )
        # Discard one card
        c2 = _card("3H")
        tracker.on_action(
            Action(c2, None, ActionType.DEAD_CARD_DISCARD), TeamId.TEAM_1
        )
        # Sync hand with 2 cards
        tracker.sync_hand([_card("4D"), _card("5C")])

        pool = tracker.get_unknown_card_pool()
        assert len(pool) == 104 - 1 - 1 - 2  # 100


# ---------------------------------------------------------------------------
# Dead position detection
# ---------------------------------------------------------------------------


class TestDeadPositions:
    def test_empty_board_no_dead_positions(self):
        tracker = _make_tracker()
        board = Board()
        # No cards used → nothing is dead
        for pos in CARD_TO_POSITIONS[_card("2S")]:
            assert not tracker.is_position_permanently_dead(pos, board.chips)

    def test_position_dead_when_all_copies_used(self):
        tracker = _make_tracker()
        card = _card("2S")
        positions = CARD_TO_POSITIONS[card]
        # Play one copy at pos[0], discard the other
        tracker.on_action(
            Action(card, positions[0], ActionType.PLACE), TeamId.TEAM_0
        )
        tracker.on_action(
            Action(card, None, ActionType.DEAD_CARD_DISCARD), TeamId.TEAM_1
        )
        # Also use up all two-eyed jacks
        for suit in (Suit.DIAMONDS, Suit.CLUBS):
            tej = Card(Rank.JACK, suit)
            tracker.on_action(
                Action(tej, Position(1, 1), ActionType.PLACE), TeamId.TEAM_0
            )
            tracker.on_action(
                Action(tej, Position(1, 2), ActionType.PLACE), TeamId.TEAM_1
            )

        board = Board()
        # pos[0] is occupied, pos[1] is empty
        board.place_chip(positions[0], TeamId.TEAM_0)

        # pos[1] should be dead — all copies of 2S gone, no two-eyed jacks left
        assert tracker.is_position_permanently_dead(positions[1], board.chips)

    def test_position_not_dead_if_two_eyed_jack_available(self):
        tracker = _make_tracker()
        card = _card("2S")
        positions = CARD_TO_POSITIONS[card]
        # Use both copies
        tracker.on_action(
            Action(card, positions[0], ActionType.PLACE), TeamId.TEAM_0
        )
        tracker.on_action(
            Action(card, None, ActionType.DEAD_CARD_DISCARD), TeamId.TEAM_1
        )
        # But two-eyed jacks still available (not exhausted)

        board = Board()
        board.place_chip(positions[0], TeamId.TEAM_0)

        # pos[1] NOT dead because two-eyed jacks still exist
        assert not tracker.is_position_permanently_dead(positions[1], board.chips)

    def test_occupied_position_not_dead(self):
        tracker = _make_tracker()
        card = _card("2S")
        pos = CARD_TO_POSITIONS[card][0]
        board = Board()
        board.place_chip(pos, TeamId.TEAM_0)

        # Occupied positions are not considered "dead"
        assert not tracker.is_position_permanently_dead(pos, board.chips)


# ---------------------------------------------------------------------------
# Guaranteed positions
# ---------------------------------------------------------------------------


class TestGuaranteedPositions:
    def test_holding_both_copies_is_guaranteed(self):
        tracker = _make_tracker()
        card = _card("7D")
        hand = [card, card]  # Both copies
        tracker.sync_hand(hand)
        board = Board()

        guaranteed = tracker.get_guaranteed_positions(hand, board.chips)
        positions = CARD_TO_POSITIONS[card]
        for pos in positions:
            assert pos in guaranteed

    def test_holding_one_copy_not_guaranteed(self):
        tracker = _make_tracker()
        card = _card("7D")
        hand = [card]  # Only one copy
        tracker.sync_hand(hand)
        board = Board()

        guaranteed = tracker.get_guaranteed_positions(hand, board.chips)
        # With 2 total copies and only holding 1, not guaranteed
        positions = CARD_TO_POSITIONS[card]
        for pos in positions:
            assert pos not in guaranteed

    def test_holding_last_copy_after_play_is_guaranteed(self):
        tracker = _make_tracker()
        card = _card("8H")
        positions = CARD_TO_POSITIONS[card]
        # One copy played
        tracker.on_action(
            Action(card, positions[0], ActionType.PLACE), TeamId.TEAM_1
        )
        board = Board()
        board.place_chip(positions[0], TeamId.TEAM_1)

        hand = [card]  # We hold the last copy
        tracker.sync_hand(hand)

        guaranteed = tracker.get_guaranteed_positions(hand, board.chips)
        # pos[0] is occupied, pos[1] should be guaranteed
        assert positions[1] in guaranteed

    def test_jacks_excluded_from_guaranteed(self):
        tracker = _make_tracker()
        jd = _card("JD")
        hand = [jd, jd]
        tracker.sync_hand(hand)
        board = Board()

        guaranteed = tracker.get_guaranteed_positions(hand, board.chips)
        # Jacks don't map to specific positions
        assert len(guaranteed) == 0


# ---------------------------------------------------------------------------
# Opponent probability
# ---------------------------------------------------------------------------


class TestOpponentProbability:
    def test_all_copies_used_zero_probability(self):
        tracker = _make_tracker()
        card = _card("2S")
        positions = CARD_TO_POSITIONS[card]
        tracker.on_action(
            Action(card, positions[0], ActionType.PLACE), TeamId.TEAM_0
        )
        tracker.on_action(
            Action(card, positions[1], ActionType.PLACE), TeamId.TEAM_1
        )

        assert tracker.opponent_has_card_probability(card) == 0.0

    def test_remaining_copies_nonzero_probability(self):
        tracker = _make_tracker()
        card = _card("2S")
        prob = tracker.opponent_has_card_probability(card)
        assert prob > 0.0
        assert prob <= 1.0

    def test_more_copies_higher_probability(self):
        tracker = _make_tracker()
        card = _card("3H")
        prob_full = tracker.opponent_has_card_probability(card)

        # Use one copy
        tracker.on_action(
            Action(card, CARD_TO_POSITIONS[card][0], ActionType.PLACE),
            TeamId.TEAM_0,
        )
        prob_after = tracker.opponent_has_card_probability(card)

        assert prob_after < prob_full
