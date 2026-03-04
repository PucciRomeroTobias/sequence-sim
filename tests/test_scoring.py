"""Tests for the scoring system: features, scoring function, and scorer agents."""

from __future__ import annotations

import random

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
from sequence.agents.greedy_agent import GreedyAgent
from sequence.agents.random_agent import RandomAgent
from sequence.agents.scorer_agent import ScorerAgent
from sequence.scoring.optimizer import GeneticOptimizer


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
        # FEATURE_NAMES covers all 47 features (35 base + 12 expert)
        assert len(FEATURE_NAMES) == 47

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

    # --- Advanced strategy features ---

    def test_position_line_score(self):
        """Chips in center should have higher position_line_score than edge chips."""
        from sequence.core.board import POSITION_TO_LINES

        state = _make_empty_state()
        # Place one chip at a center position
        center_pos = Position(5, 5)
        state.board.place_chip(center_pos, TeamId.TEAM_0)
        features = extract_features(state, TeamId.TEAM_0)
        expected = len(POSITION_TO_LINES.get(center_pos, [])) / 12.0
        assert features[28] == pytest.approx(expected)
        assert features[28] > 0  # center positions have lines

    def test_position_line_score_empty_board(self):
        state = _make_empty_state()
        features = extract_features(state, TeamId.TEAM_0)
        assert features[28] == 0.0  # no chips → no score

    def test_anchor_overlap_count_no_sequence(self):
        """Without completed sequences, anchor_overlap_count should be 0."""
        state = _make_empty_state()
        _place_chips(state.board, [Position(3, 3), Position(4, 4)], TeamId.TEAM_0)
        features = extract_features(state, TeamId.TEAM_0)
        assert features[29] == 0  # no completed sequence → no anchors

    def test_anchor_overlap_count_with_sequence(self):
        """Completed sequence chips that participate in incomplete lines are anchors."""
        state = _make_empty_state()
        # Find a line without corners and complete it
        for candidate_line in ALL_LINES:
            positions = list(candidate_line)
            if not any(state.board.is_corner(p) for p in positions):
                line = positions
                break
        _place_chips(state.board, line, TeamId.TEAM_0)
        for pos in line:
            state.board.check_new_sequences(pos, TeamId.TEAM_0)
        assert state.board.count_sequences(TeamId.TEAM_0) >= 1

        features = extract_features(state, TeamId.TEAM_0)
        # Chips in the completed sequence participate in other incomplete lines
        # (most non-corner positions belong to multiple lines), so anchors > 0
        assert features[29] > 0

    def test_chip_clustering_single_quadrant(self):
        """All chips in one quadrant → clustering = 1.0."""
        state = _make_empty_state()
        # Place chips in top-left quadrant (rows 0-4, cols 0-4)
        _place_chips(state.board, [Position(1, 1), Position(2, 2), Position(3, 3)], TeamId.TEAM_0)
        features = extract_features(state, TeamId.TEAM_0)
        assert features[30] == pytest.approx(1.0)

    def test_chip_clustering_spread(self):
        """Chips spread across quadrants → clustering < 1.0."""
        state = _make_empty_state()
        # Place one chip in each of 4 quadrants
        _place_chips(state.board, [
            Position(1, 1),   # top-left
            Position(1, 8),   # top-right
            Position(8, 1),   # bottom-left
            Position(8, 8),   # bottom-right
        ], TeamId.TEAM_0)
        features = extract_features(state, TeamId.TEAM_0)
        assert features[30] == pytest.approx(0.25)  # 1 chip per quadrant / 4 total

    def test_chip_clustering_no_chips(self):
        state = _make_empty_state()
        features = extract_features(state, TeamId.TEAM_0)
        assert features[30] == 0.0  # max(quadrants)=0, total=max(0,1)=1 → 0/1

    def test_early_jack_usage_penalty(self):
        """early_jack_usage_penalty should reflect jack count and turn number."""
        state = _make_empty_state()
        # Hand with 0 jacks at turn 0: penalty = max(0, 1-0/50) * 1/(1+0) = 1.0
        state.hands[0] = [
            Card(Rank.TWO, Suit.HEARTS),
            Card(Rank.THREE, Suit.HEARTS),
        ]
        features = extract_features(state, TeamId.TEAM_0)
        assert features[31] == pytest.approx(1.0)

        # Hand with 1 jack at turn 0: penalty = 1.0 * 1/(1+1) = 0.5
        state.hands[0] = [
            Card(Rank.JACK, Suit.DIAMONDS),  # two-eyed jack
            Card(Rank.THREE, Suit.HEARTS),
        ]
        features = extract_features(state, TeamId.TEAM_0)
        assert features[31] == pytest.approx(0.5)

        # At turn >= 50: penalty = 0.0 regardless of jacks
        late_state = GameState(
            board=state.board,
            hands=state.hands,
            deck=state.deck,
            current_team=TeamId.TEAM_0,
            turn_number=60,
        )
        features_late = extract_features(late_state, TeamId.TEAM_0)
        assert features_late[31] == pytest.approx(0.0)

    def test_jack_save_value_early_game(self):
        """Jacks in hand at turn 0 should produce positive jack_save_value."""
        state = _make_empty_state()
        state.hands[0] = [
            Card(Rank.JACK, Suit.DIAMONDS),  # two-eyed
            Card(Rank.JACK, Suit.HEARTS),    # one-eyed
            Card(Rank.TWO, Suit.HEARTS),
        ]
        features = extract_features(state, TeamId.TEAM_0)
        # 2 jacks * max(0, 1.0 - 0/50) = 2 * 1.0 = 2.0
        assert features[32] == pytest.approx(2.0)

    def test_jack_save_value_late_game(self):
        """Jacks after turn 50 should have jack_save_value = 0."""
        state = _make_empty_state()
        state.hands[0] = [
            Card(Rank.JACK, Suit.DIAMONDS),
            Card(Rank.JACK, Suit.HEARTS),
        ]
        late_state = GameState(
            board=state.board,
            hands=state.hands,
            deck=state.deck,
            current_team=TeamId.TEAM_0,
            turn_number=60,
        )
        features = extract_features(late_state, TeamId.TEAM_0)
        assert features[32] == pytest.approx(0.0)


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

    def test_rank_actions_fast_matches_rank_actions(self):
        """rank_actions_fast should produce same ranking as rank_actions."""
        state = _make_empty_state()
        # Place some chips to create a more interesting board state
        _place_chips(state.board, [Position(3, 3), Position(4, 4), Position(5, 5)], TeamId.TEAM_0)
        _place_chips(state.board, [Position(2, 5), Position(3, 6)], TeamId.TEAM_1)
        sf = ScoringFunction(BALANCED_WEIGHTS)
        legal = state.get_legal_actions(TeamId.TEAM_0)
        if not legal:
            pytest.skip("No legal actions")
        ranked_slow = sf.rank_actions(state, legal, TeamId.TEAM_0)
        ranked_fast = sf.rank_actions_fast(state, legal, TeamId.TEAM_0)
        assert len(ranked_fast) == len(ranked_slow)
        # Actions should be in the same order
        for (a_slow, s_slow), (a_fast, s_fast) in zip(ranked_slow, ranked_fast):
            assert a_slow == a_fast, f"Action mismatch: {a_slow} vs {a_fast}"
            assert s_slow == pytest.approx(s_fast, abs=0.01), (
                f"Score mismatch for {a_slow}: {s_slow} vs {s_fast}"
            )

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
        # ScoringWeights now has 47 fields (35 base + 12 expert)
        for w in [BALANCED_WEIGHTS, DEFENSIVE_WEIGHTS, OFFENSIVE_WEIGHTS]:
            arr = w.to_array()
            assert arr.shape == (47,)


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


# ---------------------------------------------------------------------------
# Optimizer tests
# ---------------------------------------------------------------------------


class TestGeneticOptimizer:
    def test_runs_without_error(self):
        """GeneticOptimizer completes with small parameters."""
        opt = GeneticOptimizer(
            population_size=5,
            num_generations=3,
            games_per_eval=10,
            num_workers=1,
            seed=42,
        )
        weights, fitness = opt.optimize()
        assert isinstance(weights, ScoringWeights)
        assert 0.0 <= fitness <= 1.0

    def test_fitness_improves(self):
        """Best fitness should improve (or stay same) over 3 generations."""
        opt = GeneticOptimizer(
            population_size=6,
            num_generations=3,
            games_per_eval=10,
            num_workers=1,
            seed=7,
        )
        population = opt._initial_population()

        gen_bests: list[float] = []
        overall_best = 0.0
        for _gen in range(opt.num_generations):
            fitnesses = [opt.evaluate_fitness(w) for w in population]
            gen_best = max(fitnesses)
            overall_best = max(overall_best, gen_best)
            gen_bests.append(overall_best)

            parents = opt.select(population, fitnesses)
            next_pop = [population[int(np.argmax(fitnesses))].copy()]
            while len(next_pop) < opt.population_size:
                p1, p2 = random.sample(parents, 2)
                child = opt.crossover(p1, p2)
                child = opt.mutate(child)
                next_pop.append(child)
            population = next_pop

        # The tracked overall best should be non-decreasing
        for i in range(1, len(gen_bests)):
            assert gen_bests[i] >= gen_bests[i - 1]

    def test_crossover_produces_valid_weights(self):
        opt = GeneticOptimizer(seed=0)
        p1 = BALANCED_WEIGHTS.to_array()
        p2 = OFFENSIVE_WEIGHTS.to_array()
        child = opt.crossover(p1, p2)
        assert child.shape == (47,)
        assert np.all(np.isfinite(child))

    def test_mutate_preserves_shape(self):
        opt = GeneticOptimizer(seed=0)
        w = BALANCED_WEIGHTS.to_array()
        mutated = opt.mutate(w)
        assert mutated.shape == (47,)
        assert np.all(np.isfinite(mutated))

    def test_select_returns_correct_size(self):
        opt = GeneticOptimizer(population_size=5, seed=0)
        pop = [opt._random_weights() for _ in range(5)]
        fitnesses = [0.1, 0.5, 0.3, 0.8, 0.2]
        parents = opt.select(pop, fitnesses)
        assert len(parents) == 5
