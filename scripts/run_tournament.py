#!/usr/bin/env python3
"""Run a round-robin tournament between Sequence agents."""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sequence.core.game import GameConfig
from sequence.agents.random_agent import RandomAgent
from sequence.agents.greedy_agent import GreedyAgent
from sequence.agents.scorer_agent import ScorerAgent
from sequence.agents.defensive_agent import DefensiveAgent
from sequence.agents.offensive_agent import OffensiveAgent
from sequence.agents.lookahead_agent import LookaheadAgent
from sequence.scoring.scoring_function import ScoringFunction, BALANCED_WEIGHTS
from sequence.simulation.tournament import Tournament


AGENT_REGISTRY = {
    "random": lambda: RandomAgent(),
    "greedy": lambda: GreedyAgent(),
    "scorer": lambda: ScorerAgent(ScoringFunction(BALANCED_WEIGHTS)),
    "defensive": lambda: DefensiveAgent(),
    "offensive": lambda: OffensiveAgent(),
    "lookahead1": lambda: LookaheadAgent(depth=1),
    "lookahead2": lambda: LookaheadAgent(depth=2),
}


def main():
    parser = argparse.ArgumentParser(description="Run a Sequence tournament")
    parser.add_argument(
        "--agents",
        type=str,
        default="random,greedy,scorer",
        help="Comma-separated agent names",
    )
    parser.add_argument("--games", type=int, default=200, help="Games per matchup")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers")
    parser.add_argument("--max-turns", type=int, default=300, help="Max turns per game")
    args = parser.parse_args()

    agent_names = [n.strip() for n in args.agents.split(",")]
    for name in agent_names:
        if name not in AGENT_REGISTRY:
            print(f"Unknown agent: {name}. Available: {list(AGENT_REGISTRY.keys())}")
            sys.exit(1)

    print(f"Tournament: {agent_names}")
    print(f"Games per matchup: {args.games}, Workers: {args.workers}")
    print()

    # Round-robin
    results: dict[str, dict[str, int]] = {n: {m: 0 for m in agent_names} for n in agent_names}
    total_games: dict[str, dict[str, int]] = {n: {m: 0 for m in agent_names} for n in agent_names}

    for i, name_a in enumerate(agent_names):
        for name_b in agent_names[i + 1 :]:
            print(f"  {name_a} vs {name_b}...", end=" ", flush=True)
            t0 = time.time()

            config = GameConfig(max_turns=args.max_turns)
            tournament = Tournament(
                agent_factories=[AGENT_REGISTRY[name_a], AGENT_REGISTRY[name_b]],
                num_games=args.games,
                config=config,
                swap_sides=True,
                max_workers=args.workers,
                show_progress=True,
            )
            result = tournament.run()

            wins_a = sum(1 for r in result.records if r.winner == 0)
            wins_b = sum(1 for r in result.records if r.winner == 1)
            draws = len(result.records) - wins_a - wins_b

            results[name_a][name_b] = wins_a
            results[name_b][name_a] = wins_b
            total_games[name_a][name_b] = len(result.records)
            total_games[name_b][name_a] = len(result.records)

            elapsed = time.time() - t0
            print(
                f"{wins_a}-{wins_b} (draws: {draws}) [{elapsed:.1f}s]"
            )

    # Print win rate matrix
    print("\n=== Win Rate Matrix ===")
    header = f"{'':>12}" + "".join(f"{n:>12}" for n in agent_names)
    print(header)
    for name in agent_names:
        row = f"{name:>12}"
        for opp in agent_names:
            if name == opp:
                row += f"{'---':>12}"
            elif total_games[name][opp] > 0:
                rate = results[name][opp] / total_games[name][opp]
                row += f"{rate:>11.1%} "
            else:
                row += f"{'N/A':>12}"
        print(row)

    # Total wins
    print("\n=== Total Wins ===")
    for name in agent_names:
        total_w = sum(results[name][o] for o in agent_names if o != name)
        total_g = sum(total_games[name][o] for o in agent_names if o != name)
        if total_g > 0:
            print(f"  {name}: {total_w}/{total_g} ({total_w / total_g:.1%})")


if __name__ == "__main__":
    main()
