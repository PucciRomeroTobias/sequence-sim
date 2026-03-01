"""Tests for GameState and Game."""

import random

from sequence.core.actions import Action, ActionType
from sequence.core.board import Board
from sequence.core.card import Card
from sequence.core.deck import Deck
from sequence.core.game import Game, GameConfig
from sequence.core.game_state import GameState
from sequence.core.types import Position, Rank, Suit, TeamId


def _make_state(seed=42) -> GameState:
    """Create a fresh game state for testing."""
    board = Board()
    deck = Deck(seed=seed)
    hands: dict[int, list[Card]] = {}
    for t in range(2):
        hand = [deck.draw() for _ in range(7)]
        hands[t] = [c for c in hand if c is not None]
    return GameState(board=board, hands=hands, deck=deck, current_team=TeamId.TEAM_0)


class _RandomAgent:
    """Minimal random agent for testing."""

    def __init__(self, seed=None):
        self._rng = random.Random(seed)

    def choose_action(self, state, legal_actions):
        return self._rng.choice(legal_actions)

    def notify_game_start(self, team, config):
        pass

    def notify_action(self, action, team):
        pass


def test_legal_actions_initial():
    state = _make_state()
    actions = state.get_legal_actions()
    assert len(actions) > 0
    # All actions should be valid types
    for a in actions:
        assert a.action_type in (
            ActionType.PLACE,
            ActionType.REMOVE,
            ActionType.DEAD_CARD_DISCARD,
        )


def test_legal_actions_place():
    state = _make_state()
    actions = state.get_legal_actions()
    place_actions = [a for a in actions if a.action_type == ActionType.PLACE]
    # Should have some placement actions
    assert len(place_actions) > 0
    for a in place_actions:
        assert a.position is not None
        assert state.board.is_empty(a.position)


def test_apply_action_place():
    state = _make_state()
    actions = state.get_legal_actions()
    place_actions = [a for a in actions if a.action_type == ActionType.PLACE]
    assert len(place_actions) > 0

    action = place_actions[0]
    new_state = state.apply_action(action)

    # Chip should be placed
    assert not new_state.board.is_empty(action.position)
    # Turn should advance
    assert new_state.current_team != state.current_team
    assert new_state.turn_number == state.turn_number + 1
    # Original state unchanged
    assert state.board.is_empty(action.position)


def test_dead_card_detection():
    """A card with both board positions occupied should become a dead card."""
    state = _make_state(seed=1)
    # Find a normal card in hand
    hand = state.hands[0]
    normal_cards = [c for c in hand if not c.is_jack]
    if not normal_cards:
        return  # Skip if no normal cards

    from sequence.core.board import CARD_TO_POSITIONS

    card = normal_cards[0]
    positions = CARD_TO_POSITIONS.get(card, [])
    if len(positions) < 2:
        return

    # Occupy both positions
    for pos in positions:
        state.board.place_chip(pos, TeamId.TEAM_1)
        state.board._empty_positions.discard(pos)

    actions = state.get_legal_actions()
    dead_actions = [
        a
        for a in actions
        if a.card == card and a.action_type == ActionType.DEAD_CARD_DISCARD
    ]
    assert len(dead_actions) >= 1


def test_is_terminal_no_winner():
    state = _make_state()
    assert state.is_terminal() is None


def test_full_game_completes():
    """A full game with random agents should terminate."""
    config = GameConfig(seed=42, max_turns=500)
    seed_val = 42

    game = Game(
        agent_factories=[lambda: _RandomAgent(seed=None)] * 2,
        config=config,
    )
    record = game.play()
    assert record.total_turns > 0
    assert record.total_turns <= 500
    assert len(record.moves) == record.total_turns


def test_deterministic_game():
    """Same seed should produce the same game."""

    def make_agent_factory(s):
        return lambda: _RandomAgent(seed=s)

    config1 = GameConfig(seed=100, max_turns=200)
    game1 = Game(
        agent_factories=[make_agent_factory(1), make_agent_factory(2)],
        config=config1,
    )
    record1 = game1.play()

    config2 = GameConfig(seed=100, max_turns=200)
    game2 = Game(
        agent_factories=[make_agent_factory(1), make_agent_factory(2)],
        config=config2,
    )
    record2 = game2.play()

    assert record1.total_turns == record2.total_turns
    assert record1.winner == record2.winner
    for m1, m2 in zip(record1.moves, record2.moves):
        assert m1.action == m2.action


def test_game_record_serialization():
    config = GameConfig(seed=42, max_turns=50)
    game = Game(
        agent_factories=[lambda: _RandomAgent(seed=10)] * 2,
        config=config,
    )
    record = game.play()
    d = record.to_dict()
    assert "game_id" in d
    assert "moves" in d
    assert isinstance(d["moves"], list)
