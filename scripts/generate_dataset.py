#!/usr/bin/env python3
"""Generate a dataset of MCTS vs MCTS games."""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sequence.core.game import GameConfig
from sequence.agents.mcts_agent import MCTSAgent
from sequence.simulation.tournament import Tournament
from sequence.simulation.dataset import DatasetWriter


def main():
    parser = argparse.ArgumentParser(description="Generate MCTS vs MCTS dataset")
    parser.add_argument("--games", type=int, default=100, help="Number of games")
    parser.add_argument(
        "--mcts-iterations", type=int, default=500, help="MCTS iterations per move"
    )
    parser.add_argument(
        "--determinizations", type=int, default=10, help="MCTS determinizations"
    )
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers")
    parser.add_argument(
        "--output",
        type=str,
        default="data/datasets/mcts_vs_mcts.jsonl",
        help="Output file",
    )
    parser.add_argument("--max-turns", type=int, default=300, help="Max turns per game")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.games} games: MCTS({args.mcts_iterations}) vs MCTS({args.mcts_iterations})")
    print(f"Determinizations: {args.determinizations}, Workers: {args.workers}")
    print(f"Output: {output_path}")
    print()

    def make_mcts():
        return MCTSAgent(
            iterations=args.mcts_iterations,
            num_determinizations=args.determinizations,
            rollout_depth=30,
        )

    config = GameConfig(max_turns=args.max_turns)
    tournament = Tournament(
        agent_factories=[make_mcts, make_mcts],
        num_games=args.games,
        config=config,
        swap_sides=False,
        max_workers=args.workers,
        show_progress=True,
    )

    t0 = time.time()
    result = tournament.run()
    elapsed = time.time() - t0
    records = result.records

    # Write to JSONL
    writer = DatasetWriter(str(output_path))
    for record in records:
        writer.write(record)
    writer.close()

    # Stats
    total_moves = sum(r.total_turns for r in records)
    team0_wins = sum(1 for r in records if r.winner == 0)
    team1_wins = sum(1 for r in records if r.winner == 1)
    draws = len(records) - team0_wins - team1_wins

    print(f"\nDone in {elapsed:.1f}s")
    print(f"Games: {len(records)}, Total moves: {total_moves}")
    print(f"Team 0 wins: {team0_wins}, Team 1 wins: {team1_wins}, Draws: {draws}")
    print(f"Avg game length: {total_moves / len(records):.1f} turns")
    print(f"Dataset written to: {output_path}")


if __name__ == "__main__":
    main()
