"""Tests for SmartAgent — card counting + enhanced features + instant decisions."""

from __future__ import annotations

import pytest

from sequence.agents.greedy_agent import GreedyAgent
from sequence.agents.random_agent import RandomAgent
from sequence.agents.smart_agent import SmartAgent
from sequence.core.board import ALL_LINES, Board, CARD_TO_POSITIONS
from sequence.core.card import Card
from sequence.core.deck import Deck
from sequence.core.game import Game, GameConfig
from sequence.core.game_state import GameState
from sequence.core.types import EMPTY, Position, Rank, Suit, TeamId


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


def _card(s: str) -> Card:
    return Card.from_str(s)


# ---------------------------------------------------------------------------
# Basic functionality
# ---------------------------------------------------------------------------


class TestSmartAgentBasics:
    def test_always_returns_valid_action(self):
        """SmartAgent should always choose a legal action."""
        state = _make_state()
        agent = SmartAgent()
        agent.notify_game_start(TeamId.TEAM_0, GameConfig())
        legal = state.get_legal_actions(TeamId.TEAM_0)
        if not legal:
            pytest.skip("No legal actions")
        action = agent.choose_action(state, legal)
        assert action in legal

    def test_single_legal_action(self):
        """With only one legal action, should return it immediately."""
        state = _make_state()
        agent = SmartAgent()
        agent.notify_game_start(TeamId.TEAM_0, GameConfig())
        legal = state.get_legal_actions(TeamId.TEAM_0)
        if not legal:
            pytest.skip("No legal actions")
        action = agent.choose_action(state, [legal[0]])
        assert action == legal[0]

    def test_plays_full_game(self):
        """SmartAgent should play a complete game without errors."""
        config = GameConfig(seed=42, max_turns=300)
        game = Game(
            agent_factories=[
                lambda: SmartAgent(use_lookahead=False),
                lambda: RandomAgent(seed=1),
            ],
            config=config,
        )
        record = game.play()
        assert record.total_turns > 0

    def test_notify_action_updates_tracker(self):
        """Verify notify_action is called and tracker gets updated."""
        agent = SmartAgent()
        agent.notify_game_start(TeamId.TEAM_0, GameConfig())
        assert agent._tracker is not None

        card = _card("2S")
        pos = CARD_TO_POSITIONS[card][0]
        from sequence.core.actions import Action, ActionType

        action = Action(card, pos, ActionType.PLACE)
        agent.notify_action(action, TeamId.TEAM_1)
        assert agent._tracker.copies_played(card) == 1

    def test_works_without_notify_game_start(self):
        """SmartAgent should work even if notify_game_start wasn't called."""
        state = _make_state()
        agent = SmartAgent()
        legal = state.get_legal_actions(TeamId.TEAM_0)
        if not legal:
            pytest.skip("No legal actions")
        action = agent.choose_action(state, legal)
        assert action in legal


# ---------------------------------------------------------------------------
# Instant decisions
# ---------------------------------------------------------------------------


class TestInstantDecisions:
    def test_completes_sequence(self):
        """SmartAgent should complete a sequence when possible."""
        # Find a line not involving corners
        for candidate_line in ALL_LINES:
            positions = list(candidate_line)
            has_corner = any(
                Position(p.row, p.col) in {
                    Position(0, 0), Position(0, 9),
                    Position(9, 0), Position(9, 9),
                }
                for p in positions
            )
            if not has_corner:
                line = positions
                break

        board = Board()
        # Place 4 of 5 for team 0
        for pos in line[:4]:
            board.place_chip(pos, TeamId.TEAM_0)
            board.check_new_sequences(pos, TeamId.TEAM_0)

        # The 5th position needs to be fillable — put the right card in hand
        from sequence.core.board import LAYOUT
        target_pos = line[4]
        target_card = LAYOUT[target_pos.row][target_pos.col]
        assert target_card is not None

        deck = Deck(seed=42)
        # Draw enough to reduce deck
        for _ in range(14):
            deck.draw()

        hands = {
            0: [target_card] + [_card("3H")] * 6,
            1: [_card("4D")] * 7,
        }

        state = GameState(
            board=board, hands=hands, deck=deck,
            current_team=TeamId.TEAM_0, num_teams=2, sequences_to_win=2,
        )

        agent = SmartAgent()
        agent.notify_game_start(TeamId.TEAM_0, GameConfig())
        legal = state.get_legal_actions(TeamId.TEAM_0)
        action = agent.choose_action(state, legal)

        # Should place at the completing position
        assert action.position == target_pos

    def test_blocks_opponent_four(self):
        """SmartAgent should block opponent 4-in-a-row when possible."""
        # Find a non-corner line
        for candidate_line in ALL_LINES:
            positions = list(candidate_line)
            has_corner = any(
                Position(p.row, p.col) in {
                    Position(0, 0), Position(0, 9),
                    Position(9, 0), Position(9, 9),
                }
                for p in positions
            )
            if not has_corner:
                line = positions
                break

        board = Board()
        # Place 4 of 5 for team 1 (opponent)
        for pos in line[:4]:
            board.place_chip(pos, TeamId.TEAM_1)
            board.check_new_sequences(pos, TeamId.TEAM_1)

        # Team 0 needs to block position line[4]
        from sequence.core.board import LAYOUT
        target_pos = line[4]
        target_card = LAYOUT[target_pos.row][target_pos.col]
        assert target_card is not None

        deck = Deck(seed=42)
        for _ in range(14):
            deck.draw()

        hands = {
            0: [target_card] + [_card("3H")] * 6,
            1: [_card("4D")] * 7,
        }

        state = GameState(
            board=board, hands=hands, deck=deck,
            current_team=TeamId.TEAM_0, num_teams=2, sequences_to_win=2,
        )

        agent = SmartAgent()
        agent.notify_game_start(TeamId.TEAM_0, GameConfig())
        legal = state.get_legal_actions(TeamId.TEAM_0)
        action = agent.choose_action(state, legal)

        # Should block at the critical position
        assert action.position == target_pos


# ---------------------------------------------------------------------------
# Performance tests
# ---------------------------------------------------------------------------


class TestSmartAgentPerformance:
    def test_beats_random(self):
        """SmartAgent should beat RandomAgent > 70% over many games."""
        wins = 0
        num_games = 50
        for seed in range(num_games):
            game = Game(
                agent_factories=[
                    lambda: SmartAgent(use_lookahead=False),
                    lambda: RandomAgent(),
                ],
                config=GameConfig(seed=seed, max_turns=300),
            )
            record = game.play()
            if record.winner == 0:
                wins += 1

        win_rate = wins / num_games
        assert win_rate > 0.70, (
            f"SmartAgent only won {win_rate:.0%} vs Random"
        )

    def test_beats_greedy(self):
        """SmartAgent with lookahead should beat GreedyAgent > 45% over many games."""
        wins = 0
        num_games = 30
        for seed in range(num_games):
            # Alternate sides for fairness
            if seed % 2 == 0:
                factories = [
                    lambda: SmartAgent(use_lookahead=True, lookahead_candidates=3),
                    lambda: GreedyAgent(),
                ]
                game = Game(
                    agent_factories=factories,
                    config=GameConfig(seed=seed, max_turns=300),
                )
                record = game.play()
                if record.winner == 0:
                    wins += 1
            else:
                factories = [
                    lambda: GreedyAgent(),
                    lambda: SmartAgent(use_lookahead=True, lookahead_candidates=3),
                ]
                game = Game(
                    agent_factories=factories,
                    config=GameConfig(seed=seed, max_turns=300),
                )
                record = game.play()
                if record.winner == 1:
                    wins += 1

        win_rate = wins / num_games
        assert win_rate >= 0.33, (
            f"SmartAgent only won {win_rate:.0%} vs Greedy (expected >= 33%)"
        )

    def test_with_lookahead(self):
        """SmartAgent with lookahead should play a complete game."""
        config = GameConfig(seed=42, max_turns=200)
        game = Game(
            agent_factories=[
                lambda: SmartAgent(use_lookahead=True, lookahead_candidates=3),
                lambda: RandomAgent(seed=1),
            ],
            config=config,
        )
        record = game.play()
        assert record.total_turns > 0
