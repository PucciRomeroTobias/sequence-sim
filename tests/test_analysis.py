"""Tests for analysis modules: statistics, heatmaps, explainer."""

from __future__ import annotations

import numpy as np
import pytest

from sequence.analysis.statistics import (
    average_game_length,
    compute_elo,
    first_player_advantage,
    print_tournament_results,
    win_rate_with_ci,
)
from sequence.analysis.heatmaps import (
    placement_frequency,
    win_contribution,
    mcts_attention,
    sequence_participation,
)
from sequence.analysis.explainer import (
    explain_weights,
    generate_report,
    rank_tips_by_importance,
)
from sequence.core.game import GameRecord, MoveRecord
from sequence.scoring.scoring_function import (
    BALANCED_WEIGHTS,
    DEFENSIVE_WEIGHTS,
    ScoringWeights,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_move(
    turn: int,
    team: int,
    position: list[int] | None = None,
    mcts_visits: dict[str, int] | None = None,
    sequences_before: dict[int, int] | None = None,
    sequences_after: dict[int, int] | None = None,
) -> MoveRecord:
    return MoveRecord(
        turn=turn,
        team=team,
        action={
            "card": "2H",
            "position": position or [0, 0],
            "action_type": "place",
        },
        legal_actions_count=5,
        hand_before=["2H", "3S"],
        board_snapshot=[],
        card_drawn="4D",
        sequences_before=sequences_before or {0: 0, 1: 0},
        sequences_after=sequences_after or {0: 0, 1: 0},
        thinking_time_ms=10.0,
        mcts_visits=mcts_visits,
        mcts_values=None,
    )


def _make_record(
    winner: int | None,
    total_turns: int = 20,
    moves: list[MoveRecord] | None = None,
) -> GameRecord:
    return GameRecord(
        game_id="test-001",
        seed=42,
        agent_names=["AgentA", "AgentB"],
        config={"num_teams": 2, "hand_size": 7, "sequences_to_win": 2, "max_turns": 500},
        winner=winner,
        total_turns=total_turns,
        moves=moves or [],
        duration_ms=100.0,
    )


# ---------------------------------------------------------------------------
# Statistics tests
# ---------------------------------------------------------------------------


class TestWilsonCI:
    def test_perfect_win_rate(self):
        rate, lower, upper = win_rate_with_ci(100, 100)
        assert rate == pytest.approx(1.0)
        assert lower > 0.95
        assert upper == pytest.approx(1.0)

    def test_zero_wins(self):
        rate, lower, upper = win_rate_with_ci(0, 100)
        assert rate == pytest.approx(0.0)
        assert lower == pytest.approx(0.0)
        assert upper < 0.05

    def test_fifty_percent(self):
        rate, lower, upper = win_rate_with_ci(50, 100)
        assert rate == pytest.approx(0.5)
        assert lower < 0.5
        assert upper > 0.5
        # CI should be symmetric around 0.5
        assert lower > 0.35
        assert upper < 0.65

    def test_empty_total(self):
        rate, lower, upper = win_rate_with_ci(0, 0)
        assert rate == 0.0
        assert lower == 0.0
        assert upper == 0.0

    def test_ci_narrows_with_more_samples(self):
        _, l1, u1 = win_rate_with_ci(5, 10)
        _, l2, u2 = win_rate_with_ci(50, 100)
        _, l3, u3 = win_rate_with_ci(500, 1000)
        # Wider CI with fewer samples
        assert (u1 - l1) > (u2 - l2) > (u3 - l3)


class TestElo:
    def test_convergence_strong_player(self):
        """A player who always wins should have higher Elo."""
        results = []
        for _ in range(100):
            results.append(("strong", "weak", 1.0))
        ratings = compute_elo(results)
        assert ratings["strong"] > ratings["weak"]
        assert ratings["strong"] > 1500  # Above initial
        assert ratings["weak"] < 1500  # Below initial

    def test_draws_keep_equal(self):
        """All draws should keep ratings close to initial."""
        results = []
        for _ in range(50):
            results.append(("a", "b", 0.5))
        ratings = compute_elo(results)
        assert abs(ratings["a"] - 1500) < 5
        assert abs(ratings["b"] - 1500) < 5

    def test_multiple_players(self):
        """Test with 3 players of different strengths."""
        results = []
        for _ in range(50):
            results.append(("best", "mid", 1.0))
            results.append(("mid", "worst", 1.0))
            results.append(("best", "worst", 1.0))
        ratings = compute_elo(results)
        assert ratings["best"] > ratings["mid"] > ratings["worst"]


class TestGameStats:
    def test_first_player_advantage(self):
        records = [
            _make_record(winner=0),
            _make_record(winner=0),
            _make_record(winner=1),
            _make_record(winner=0),
        ]
        assert first_player_advantage(records) == pytest.approx(0.75)

    def test_first_player_advantage_empty(self):
        assert first_player_advantage([]) == 0.0

    def test_average_game_length(self):
        records = [
            _make_record(winner=0, total_turns=10),
            _make_record(winner=1, total_turns=30),
        ]
        assert average_game_length(records) == pytest.approx(20.0)

    def test_average_game_length_empty(self):
        assert average_game_length([]) == 0.0


class TestTournamentResults:
    def test_format(self):
        names = ["A", "B"]
        matrix = [[0, 5], [3, 0]]
        result = print_tournament_results(names, matrix)
        assert "A" in result
        assert "B" in result
        assert "5" in result
        assert "3" in result


# ---------------------------------------------------------------------------
# Heatmap tests
# ---------------------------------------------------------------------------


class TestHeatmaps:
    def test_placement_frequency(self):
        moves = [
            _make_move(0, 0, [3, 4]),
            _make_move(1, 1, [3, 4]),
            _make_move(2, 0, [5, 5]),
        ]
        records = [_make_record(winner=0, moves=moves)]
        freq = placement_frequency(records)
        assert freq[3, 4] == 2
        assert freq[5, 5] == 1
        assert freq[0, 0] == 0

    def test_placement_frequency_team_filter(self):
        moves = [
            _make_move(0, 0, [3, 4]),
            _make_move(1, 1, [3, 4]),
        ]
        records = [_make_record(winner=0, moves=moves)]
        freq_t0 = placement_frequency(records, team=0)
        freq_t1 = placement_frequency(records, team=1)
        assert freq_t0[3, 4] == 1
        assert freq_t1[3, 4] == 1

    def test_win_contribution(self):
        moves = [
            _make_move(0, 0, [2, 2]),  # winner played here
            _make_move(1, 1, [3, 3]),  # loser played here
        ]
        records = [_make_record(winner=0, moves=moves)]
        contrib = win_contribution(records)
        assert contrib[2, 2] > 0  # winner's position
        assert contrib[3, 3] < 0  # loser's position

    def test_mcts_attention(self):
        moves = [
            _make_move(0, 0, [1, 1], mcts_visits={"1,1": 100, "2,2": 50}),
            _make_move(1, 1, [3, 3], mcts_visits={"3,3": 80}),
        ]
        records = [_make_record(winner=0, moves=moves)]
        attn = mcts_attention(records)
        assert attn[1, 1] > 0
        assert attn[2, 2] > 0
        assert attn[3, 3] > 0

    def test_mcts_attention_no_data(self):
        moves = [_make_move(0, 0, [1, 1])]
        records = [_make_record(winner=0, moves=moves)]
        attn = mcts_attention(records)
        assert attn.sum() == 0

    def test_sequence_participation(self):
        moves = [
            _make_move(
                0, 0, [4, 4],
                sequences_before={0: 0, 1: 0},
                sequences_after={0: 1, 1: 0},
            ),
        ]
        records = [_make_record(winner=0, moves=moves)]
        part = sequence_participation(records)
        assert part[4, 4] == 1


# ---------------------------------------------------------------------------
# Explainer tests
# ---------------------------------------------------------------------------


class TestExplainer:
    def test_explain_weights_returns_tips(self):
        tips = explain_weights(BALANCED_WEIGHTS)
        assert len(tips) >= 5

    def test_explain_weights_sorted_by_magnitude(self):
        ranked = rank_tips_by_importance(BALANCED_WEIGHTS)
        magnitudes = [abs(v) for _, v in ranked]
        assert magnitudes == sorted(magnitudes, reverse=True)

    def test_explain_defensive_weights(self):
        tips = explain_weights(DEFENSIVE_WEIGHTS)
        # Defensive weights should mention blocking
        assert any("block" in t.lower() or "avoid" in t.lower() for t in tips)

    def test_rank_tips_includes_all_features(self):
        ranked = rank_tips_by_importance(BALANCED_WEIGHTS)
        names = [name for name, _ in ranked]
        assert len(names) == 17
        assert "completed_sequences" in names

    def test_generate_report_without_records(self):
        report = generate_report(BALANCED_WEIGHTS)
        assert "STRATEGY REPORT" in report
        assert "Weight Analysis" in report
        assert "Top 5 Priorities" in report

    def test_generate_report_with_records(self):
        records = [
            _make_record(winner=0, total_turns=20),
            _make_record(winner=1, total_turns=30),
        ]
        report = generate_report(BALANCED_WEIGHTS, records=records)
        assert "Game Statistics" in report
        assert "Games analyzed: 2" in report

    def test_comparison_center_vs_corner(self):
        # Create weights where center is much higher than corner
        weights = ScoringWeights(center_control=10.0, corner_adjacency=2.0)
        tips = explain_weights(weights)
        assert any("center control" in t.lower() for t in tips)
