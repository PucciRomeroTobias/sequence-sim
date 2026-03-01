"""Tests for the scoring system: features, scoring function, and scorer agents."""

from __future__ import annotations

import numpy as np
import pytest

from sequence.core.board import ALL_LINES, Board
from sequence.core.card import Card
from sequence.core.deck import Deck
from sequence.core.game import Game, GameConfig
from sequence.core.game_state import GameState
from sequence.core.types import CORNER, EMPTY, Position, Rank, Suit, TeamId
from sequence.scoring.features import NUM_FEATURES, extract_features
from sequence.scoring.scoring_function import (
    BALANCED_WEIGHTS,
    DEFENSIVE_WEIGHTS,
    FEATURE_NAMES,
    OFFENSIVE_WEIGHTS,
    ScoringFunction,
    ScoringWeights,
)
from sequence.agents.random_agent import RandomAgent
from sequence.agents.scorer_agent import ScorerAgent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_empty_state(seed: int = 42) -> GameState:
    """Create a fresh game state with dealt hands."""
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
        board=board,
        hands=hands,
        deck=deck,
        current_team=TeamId.TEAM_0,
        num_teams=2,
        sequences_to_win=2,
    )


def _place_chips(board: Board, positions: list[Position], team: TeamId) -> None:
    """Place chips at given positions for a team."""
    for pos in positions:
        board.place_chip(pos, team)


# ---------------------------------------------------------------------------
# Feature extraction tests
# ---------------------------------------------------------------------------


class TestFeatureExtraction:
    def test_feature_count(self):
        state = _make_empty_state()
        features = extract_features(state, TeamId.TEAM_0)
        assert features.shape == (NUM_FEATURES,)
        assert len(FEATURE_NAMES) == NUM_FEATURES

    def test_empty_board_features(self):
        state = _make_empty_state()
        features = extract_features(state, TeamId.TEAM_0)
        # No chips on board -> no sequences, no chips count
        assert features[0] == 0  # completed_sequences
        assert features[1] == 0  # four_in_a_row
        assert features[7] == 0  # chips_on_board
        assert features[8] == 0  # opp_chips_on_board

    def test_chips_on_board_counted(self):
        state = _make_empty_state()
        # Place some chips for team 0
        positions_t0 = [Position(1, 1), Position(2, 2), Position(3, 3)]
        _place_chips(state.board, positions_t0, TeamId.TEAM_0)
        # Place some chips for team 1
        positions_t1 = [Position(5, 5), Position(6, 6)]
        _place_chips(state.board, positions_t1, TeamId.TEAM_1)

        features = extract_features(state, TeamId.TEAM_0)
        assert features[7] == 3  # chips_on_board
        assert features[8] == 2  # opp_chips_on_board

    def test_center_control(self):
        state = _make_empty_state()
        # Place chips in center (rows/cols 2-7)
        center_pos = [Position(3, 3), Position(4, 4), Position(5, 5)]
        _place_chips(state.board, center_pos, TeamId.TEAM_0)
        # Place chip outside center
        state.board.place_chip(Position(0, 1), TeamId.TEAM_0)

        features = extract_features(state, TeamId.TEAM_0)
        assert features[9] == 3  # center_control (only center chips)
        assert features[7] == 4  # chips_on_board (all chips)

    def test_completed_sequence_detected(self):
        """Place 5 in a row and verify completed_sequences is counted."""
        state = _make_empty_state()
        # Use first horizontal line: positions (0,1) through (0,5)
        # But (0,0) is a corner, so use row 1 instead
        line = ALL_LINES[0]  # First horizontal line
        # Find a line that doesn't involve corners
        for candidate_line in ALL_LINES:
            positions = list(candidate_line)
            has_corner = any(
                state.board.is_corner(p) for p in positions
            )
            if not has_corner:
                line = candidate_line
                break

        positions = list(line)
        _place_chips(state.board, positions, TeamId.TEAM_0)
        # Trigger sequence detection
        for pos in positions:
            state.board.check_new_sequences(pos, TeamId.TEAM_0)

        features = extract_features(state, TeamId.TEAM_0)
        assert features[0] >= 1  # completed_sequences

    def test_four_in_a_row(self):
        """Place 4 chips in a line with no opponent -> four_in_a_row counted."""
        state = _make_empty_state()
        # Use a horizontal line, place only 4 of 5
        # Row 1: positions (1,0) through (1,4)
        line = None
        for candidate in ALL_LINES:
            positions = list(candidate)
            has_corner = any(state.board.is_corner(p) for p in positions)
            if not has_corner:
                line = positions
                break
        assert line is not None
        # Place first 4
        _place_chips(state.board, line[:4], TeamId.TEAM_0)

        features = extract_features(state, TeamId.TEAM_0)
        assert features[1] >= 1  # four_in_a_row

    def test_opponent_features_mirror(self):
        """Opponent features for team 0 should match own features for team 1."""
        state = _make_empty_state()
        # Place chips for team 1
        _place_chips(state.board, [Position(3, 3), Position(4, 4)], TeamId.TEAM_1)

        features_t0 = extract_features(state, TeamId.TEAM_0)
        features_t1 = extract_features(state, TeamId.TEAM_1)
        # Team 0's opp_chips_on_board should equal Team 1's chips_on_board
        assert features_t0[8] == features_t1[7]

    def test_hand_analysis(self):
        state = _make_empty_state()
        # Override hand with known cards
        state.hands[0] = [
            Card(Rank.TWO, Suit.HEARTS),
            Card(Rank.TWO, Suit.HEARTS),  # pair
            Card(Rank.JACK, Suit.DIAMONDS),  # two-eyed jack
            Card(Rank.JACK, Suit.CLUBS),  # two-eyed jack
            Card(Rank.JACK, Suit.HEARTS),  # one-eyed jack
            Card(Rank.THREE, Suit.SPADES),
            Card(Rank.FOUR, Suit.SPADES),
        ]

        features = extract_features(state, TeamId.TEAM_0)
        assert features[11] == 1  # hand_pairs (one pair of 2H)
        assert features[12] == 2  # two_eyed_jacks_in_hand
        assert features[13] == 1  # one_eyed_jacks_in_hand

    def test_dead_cards(self):
        state = _make_empty_state()
        # Card 2S appears at two positions on the board. Fill both.
        card_2s = Card(Rank.TWO, Suit.SPADES)
        from sequence.core.board import CARD_TO_POSITIONS

        positions = CARD_TO_POSITIONS[card_2s]
        for pos in positions:
            state.board.place_chip(pos, TeamId.TEAM_1)

        # Put 2S in hand
        state.hands[0] = [card_2s]
        features = extract_features(state, TeamId.TEAM_0)
        assert features[14] == 1  # dead_cards_in_hand

    def test_corner_adjacency(self):
        state = _make_empty_state()
        # Position (1,1) is adjacent to corner (0,0)
        state.board.place_chip(Position(1, 1), TeamId.TEAM_0)
        features = extract_features(state, TeamId.TEAM_0)
        assert features[10] >= 1  # corner_adjacency


# ---------------------------------------------------------------------------
# Scoring function tests
# ---------------------------------------------------------------------------


class TestScoringFunction:
    def test_evaluate_returns_float(self):
        state = _make_empty_state()
        sf = ScoringFunction(BALANCED_WEIGHTS)
        score = sf.evaluate(state, TeamId.TEAM_0)
        assert isinstance(score, float)

    def test_rank_actions(self):
        state = _make_empty_state()
        sf = ScoringFunction(BALANCED_WEIGHTS)
        legal = state.get_legal_actions(TeamId.TEAM_0)
        if not legal:
            pytest.skip("No legal actions in initial state")
        ranked = sf.rank_actions(state, legal, TeamId.TEAM_0)
        assert len(ranked) == len(legal)
        # Check sorted descending
        scores = [s for _, s in ranked]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1]

    def test_weights_serialization(self):
        w = BALANCED_WEIGHTS
        json_str = w.to_json()
        w2 = ScoringWeights.from_json(json_str)
        assert np.allclose(w.to_array(), w2.to_array())

    def test_weights_from_array(self):
        arr = BALANCED_WEIGHTS.to_array()
        w = ScoringWeights.from_array(arr)
        assert np.allclose(w.to_array(), arr)

    def test_all_weight_sets_have_correct_size(self):
        for w in [BALANCED_WEIGHTS, DEFENSIVE_WEIGHTS, OFFENSIVE_WEIGHTS]:
            arr = w.to_array()
            assert arr.shape == (NUM_FEATURES,)


# ---------------------------------------------------------------------------
# Scorer agent tests
# ---------------------------------------------------------------------------


class TestScorerAgent:
    def test_scorer_agent_makes_valid_moves(self):
        state = _make_empty_state()
        agent = ScorerAgent()
        agent.notify_game_start(TeamId.TEAM_0, GameConfig())
        legal = state.get_legal_actions(TeamId.TEAM_0)
        if not legal:
            pytest.skip("No legal actions")
        action = agent.choose_action(state, legal)
        assert action in legal

    def test_scorer_agent_beats_random(self):
        """ScorerAgent should beat RandomAgent more than 60% over many games."""
        wins = {0: 0, 1: 0, "draw": 0}
        num_games = 50

        for seed in range(num_games):
            game = Game(
                agent_factories=[
                    lambda: ScorerAgent(BALANCED_WEIGHTS),
                    lambda: RandomAgent(seed=None),
                ],
                config=GameConfig(seed=seed, max_turns=300),
            )
            record = game.play()
            if record.winner == 0:
                wins[0] += 1
            elif record.winner == 1:
                wins[1] += 1
            else:
                wins["draw"] += 1

        scorer_win_rate = wins[0] / num_games
        assert scorer_win_rate > 0.60, (
            f"ScorerAgent only won {scorer_win_rate:.0%} "
            f"(wins={wins[0]}, losses={wins[1]}, draws={wins['draw']})"
        )
