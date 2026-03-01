#!/usr/bin/env python3
"""Generate a dataset of MCTS vs MCTS games."""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sequence.core.game import Game, GameConfig
from sequence.agents.mcts_agent import MCTSAgent
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

    mcts_iters = args.mcts_iterations
    mcts_dets = args.determinizations

    print(f"Generating {args.games} games: MCTS({mcts_iters}) vs MCTS({mcts_iters})")
    print(f"Determinizations: {mcts_dets}")
    print(f"Output: {output_path}")
    print()

    writer = DatasetWriter(str(output_path))
    total_moves = 0
    team0_wins = 0
    team1_wins = 0

    t0 = time.time()
    for i in range(args.games):
        config = GameConfig(seed=i, max_turns=args.max_turns)
        game = Game(
            agent_factories=[
                lambda: MCTSAgent(iterations=mcts_iters, num_determinizations=mcts_dets, rollout_depth=30),
                lambda: MCTSAgent(iterations=mcts_iters, num_determinizations=mcts_dets, rollout_depth=30),
            ],
            config=config,
        )
        record = game.play()
        writer.write(record)

        total_moves += record.total_turns
        if record.winner == 0:
            team0_wins += 1
        elif record.winner == 1:
            team1_wins += 1

        elapsed = time.time() - t0
        avg_per_game = elapsed / (i + 1)
        remaining = avg_per_game * (args.games - i - 1)
        print(
            f"  Game {i + 1}/{args.games}: {record.total_turns} turns, "
            f"winner=Team{record.winner} [{elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining]"
        )

    writer.close()
    elapsed = time.time() - t0
    draws = args.games - team0_wins - team1_wins

    print(f"\nDone in {elapsed:.1f}s")
    print(f"Games: {args.games}, Total moves: {total_moves}")
    print(f"Team 0 wins: {team0_wins}, Team 1 wins: {team1_wins}, Draws: {draws}")
    if args.games > 0:
        print(f"Avg game length: {total_moves / args.games:.1f} turns")
    print(f"Dataset written to: {output_path}")


if __name__ == "__main__":
    main()
