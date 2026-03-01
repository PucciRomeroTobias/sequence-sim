"""Tests for MCTS agent."""

import random

from sequence.core.board import Board
from sequence.core.card import Card, make_full_deck
from sequence.core.deck import Deck
from sequence.core.game import Game, GameConfig
from sequence.core.game_state import GameState
from sequence.core.types import Position, Rank, Suit, TeamId
from sequence.agents.mcts_agent import MCTSAgent, MCTSNode


def _make_state(seed=42):
    board = Board()
    deck = Deck(seed=seed)
    hands = {}
    for t in range(2):
        hands[t] = [deck.draw() for _ in range(7)]
    return GameState(board=board, hands=hands, deck=deck, current_team=TeamId.TEAM_0)


class _RandomAgent:
    def __init__(self, seed=None):
        self._rng = random.Random(seed)

    def choose_action(self, state, legal_actions):
        return self._rng.choice(legal_actions)

    def notify_game_start(self, team, config):
        pass

    def notify_action(self, action, team):
        pass


def test_mcts_node_ucb1():
    state = _make_state()
    root = MCTSNode(state)
    root.visits = 10

    child_state = state.copy()
    child = MCTSNode(child_state, parent=root)
    child.visits = 5
    child.total_value = 3.0

    ucb = child.ucb1(1.41)
    # exploitation = 3/5 = 0.6
    # exploration = 1.41 * sqrt(ln(10)/5) = 1.41 * sqrt(2.302/5) ≈ 1.41 * 0.679 ≈ 0.957
    assert ucb > 0.6  # At least exploitation
    assert ucb < 3.0  # Not unreasonable


def test_mcts_node_unvisited_ucb1():
    state = _make_state()
    root = MCTSNode(state)
    root.visits = 10

    child_state = state.copy()
    child = MCTSNode(child_state, parent=root)
    child.visits = 0

    assert child.ucb1(1.41) == float("inf")


def test_mcts_determinization_valid():
    """Determinization should produce a valid state."""
    state = _make_state()
    agent = MCTSAgent(iterations=1, num_determinizations=1, seed=42)
    agent._team = TeamId.TEAM_0

    det_state = agent._determinize(state)

    # Opponent hand should have the right size
    assert len(det_state.hands[1]) == 7
    # Our hand should be unchanged
    assert det_state.hands[0] == state.hands[0]


def test_mcts_1_iteration_returns_valid_action():
    """MCTS with 1 iteration should return a valid action."""
    state = _make_state()
    legal = state.get_legal_actions()
    agent = MCTSAgent(iterations=1, num_determinizations=1, seed=42)
    agent._team = TeamId.TEAM_0

    action = agent.choose_action(state, legal)
    assert action in legal


def test_mcts_stores_visit_data():
    """MCTS should store visit data for dataset generation."""
    state = _make_state()
    legal = state.get_legal_actions()
    agent = MCTSAgent(iterations=50, num_determinizations=2, seed=42)
    agent._team = TeamId.TEAM_0

    agent.choose_action(state, legal)

    assert agent.last_mcts_visits is not None
    assert isinstance(agent.last_mcts_visits, dict)
    assert len(agent.last_mcts_visits) > 0
    assert all(isinstance(v, int) for v in agent.last_mcts_visits.values())


def test_mcts_beats_random():
    """MCTS with moderate iterations should beat random agent."""
    wins = 0
    num_games = 10

    for i in range(num_games):
        config = GameConfig(seed=i * 100, max_turns=200)
        game = Game(
            agent_factories=[
                lambda: MCTSAgent(
                    iterations=100, num_determinizations=3,
                    rollout_depth=20, max_root_actions=12, seed=42
                ),
                lambda: _RandomAgent(seed=None),
            ],
            config=config,
        )
        record = game.play()
        if record.winner == 0:
            wins += 1

    win_rate = wins / num_games
    assert win_rate >= 0.5, f"MCTS win rate vs random: {win_rate:.1%}"


def test_mcts_single_action():
    """When only one action is legal, MCTS returns it immediately."""
    state = _make_state()
    legal = state.get_legal_actions()

    # Use just the first action
    agent = MCTSAgent(iterations=100, seed=42)
    agent._team = TeamId.TEAM_0

    action = agent.choose_action(state, [legal[0]])
    assert action == legal[0]
