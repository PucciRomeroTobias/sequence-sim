"""Tests for agents, simulation runner, tournament, and dataset."""

import tempfile
from pathlib import Path

from sequence.agents import GreedyAgent, RandomAgent
from sequence.core.game import Game, GameConfig, GameRecord
from sequence.simulation.dataset import DatasetReader, DatasetWriter, to_move_dataframe
from sequence.simulation.runner import run_single_game
from sequence.simulation.tournament import Tournament


# ---------------------------------------------------------------------------
# RandomAgent tests
# ---------------------------------------------------------------------------


def test_random_agent_valid_actions():
    """RandomAgent should always return a legal action."""
    config = GameConfig(seed=42, max_turns=100)
    game = Game(
        agent_factories=[lambda: RandomAgent(seed=1), lambda: RandomAgent(seed=2)],
        config=config,
    )
    record = game.play()
    assert record.total_turns > 0
    # All moves should have been recorded
    assert len(record.moves) == record.total_turns


def test_random_agent_deterministic():
    """Same seeds should produce the same game."""

    def play_game():
        config = GameConfig(seed=100, max_turns=200)
        game = Game(
            agent_factories=[lambda: RandomAgent(seed=10), lambda: RandomAgent(seed=20)],
            config=config,
        )
        return game.play()

    r1 = play_game()
    r2 = play_game()
    assert r1.total_turns == r2.total_turns
    assert r1.winner == r2.winner
    for m1, m2 in zip(r1.moves, r2.moves):
        assert m1.action == m2.action


# ---------------------------------------------------------------------------
# GreedyAgent tests
# ---------------------------------------------------------------------------


def test_greedy_agent_valid_actions():
    """GreedyAgent should always return a legal action."""
    config = GameConfig(seed=7, max_turns=200)
    game = Game(
        agent_factories=[lambda: GreedyAgent(seed=1), lambda: RandomAgent(seed=2)],
        config=config,
    )
    record = game.play()
    assert record.total_turns > 0


def test_greedy_beats_random():
    """GreedyAgent should beat RandomAgent more than 60% of the time over 100 games."""
    greedy_wins = 0
    total = 100
    for i in range(total):
        seed_val = i
        config = GameConfig(seed=seed_val, max_turns=500)
        if i % 2 == 0:
            # Greedy as team 0
            factories = [lambda: GreedyAgent(), lambda: RandomAgent()]
            game = Game(agent_factories=factories, config=config)
            record = game.play()
            if record.winner == 0:
                greedy_wins += 1
        else:
            # Greedy as team 1
            factories = [lambda: RandomAgent(), lambda: GreedyAgent()]
            game = Game(agent_factories=factories, config=config)
            record = game.play()
            if record.winner == 1:
                greedy_wins += 1

    win_rate = greedy_wins / total
    assert win_rate > 0.60, f"GreedyAgent win rate {win_rate:.2%} <= 60%"


# ---------------------------------------------------------------------------
# Simulation runner tests
# ---------------------------------------------------------------------------


def test_run_single_game():
    """run_single_game should return a valid GameRecord."""
    config = GameConfig(seed=42, max_turns=200)
    record = run_single_game(
        agent_factories=[lambda: RandomAgent(seed=1), lambda: RandomAgent(seed=2)],
        config=config,
    )
    assert isinstance(record, GameRecord)
    assert record.total_turns > 0


# ---------------------------------------------------------------------------
# Tournament tests
# ---------------------------------------------------------------------------


def test_tournament_runs():
    """Tournament should run and return results."""
    tournament = Tournament(
        agent_factories=[lambda: RandomAgent(), lambda: RandomAgent()],
        num_games=10,
        config=GameConfig(max_turns=200),
        swap_sides=True,
        max_workers=1,
        show_progress=False,
    )
    result = tournament.run()
    assert result.total_games == 10
    assert len(result.records) == 10
    # Wins + draws should equal total
    total_wins = sum(result.wins.values())
    assert total_wins + result.draws == result.total_games


# ---------------------------------------------------------------------------
# Dataset tests
# ---------------------------------------------------------------------------


def test_dataset_roundtrip():
    """Writing and reading game records via JSONL should roundtrip correctly."""
    # Generate a few game records
    records = []
    for i in range(3):
        config = GameConfig(seed=i, max_turns=50)
        record = run_single_game(
            agent_factories=[lambda: RandomAgent(seed=1), lambda: RandomAgent(seed=2)],
            config=config,
        )
        records.append(record)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_games.jsonl"

        # Write
        with DatasetWriter(path) as writer:
            writer.write_many(records)

        # Read back
        reader = DatasetReader(path)
        loaded = reader.read_all()

        assert len(loaded) == len(records)
        for orig, loaded_rec in zip(records, loaded):
            assert orig.game_id == loaded_rec.game_id
            assert orig.winner == loaded_rec.winner
            assert orig.total_turns == loaded_rec.total_turns
            assert len(orig.moves) == len(loaded_rec.moves)
            for m1, m2 in zip(orig.moves, loaded_rec.moves):
                assert m1.action == m2.action


def test_to_move_dataframe():
    """to_move_dataframe should produce a DataFrame with expected columns."""
    import pandas as pd

    config = GameConfig(seed=42, max_turns=50)
    record = run_single_game(
        agent_factories=[lambda: RandomAgent(seed=1), lambda: RandomAgent(seed=2)],
        config=config,
    )
    df = to_move_dataframe([record])
    assert isinstance(df, pd.DataFrame)
    assert len(df) == record.total_turns
    assert "game_id" in df.columns
    assert "turn" in df.columns
    assert "action_type" in df.columns
    assert "card" in df.columns
