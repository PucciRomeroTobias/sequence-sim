#!/usr/bin/env python3
"""Optimize scoring function weights using genetic algorithm."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    parser = argparse.ArgumentParser(description="Optimize scoring weights")
    parser.add_argument("--generations", type=int, default=50, help="Number of generations")
    parser.add_argument("--population", type=int, default=30, help="Population size")
    parser.add_argument(
        "--games-per-eval", type=int, default=50, help="Games per fitness evaluation"
    )
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers")
    parser.add_argument(
        "--output",
        type=str,
        default="data/weights/optimized.json",
        help="Output file for best weights",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["ga", "cmaes"],
        default="ga",
        help="Optimization method",
    )
    parser.add_argument(
        "--smart",
        action="store_true",
        help="Use SmartAgent (card tracking) instead of ScorerAgent",
    )
    parser.add_argument(
        "--mixed",
        action="store_true",
        help="Use mixed opponent pool (Lookahead2 50%%, Greedy 25%%, Defensive 25%%)",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=5.0,
        help="Initial sigma for CMA-ES (default: 5.0)",
    )
    parser.add_argument(
        "--lookahead",
        action="store_true",
        help="Enable depth-1 lookahead during evaluation (5-10x slower)",
    )
    parser.add_argument(
        "--self-play-rounds",
        type=int,
        default=1,
        help="Number of self-play rounds (each adds best from previous round as opponent)",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    from sequence.scoring.optimizer import (
        GeneticOptimizer,
        add_custom_opponent,
        clear_custom_opponents,
    )

    self_play_rounds = args.self_play_rounds
    best_weights = None
    best_fitness = -1.0

    clear_custom_opponents()

    for round_num in range(1, self_play_rounds + 1):
        if self_play_rounds > 1:
            print(f"\n{'='*60}")
            print(f"SELF-PLAY ROUND {round_num}/{self_play_rounds}")
            print(f"{'='*60}\n")

        if args.method == "ga":
            agent_type = "SmartAgent" if args.smart else "ScorerAgent"
            opp_type = "Mixed (Lookahead2/Greedy/Defensive)" if args.mixed else "GreedyAgent"
            print(f"Running Genetic Algorithm:")
            print(f"  Agent: {agent_type}")
            print(f"  Opponent: {opp_type}")
            print(f"  Generations: {args.generations}")
            print(f"  Population: {args.population}")
            print(f"  Games per eval: {args.games_per_eval}")
            print(f"  Lookahead: {args.lookahead}")
            print()

            optimizer = GeneticOptimizer(
                population_size=args.population,
                num_generations=args.generations,
                games_per_eval=args.games_per_eval,
                num_workers=args.workers,
                use_smart_agent=args.smart,
                use_mixed_opponents=args.mixed,
                use_lookahead=args.lookahead,
            )
            best_weights, best_fitness = optimizer.optimize()

        elif args.method == "cmaes":
            try:
                from sequence.scoring.optimizer import CMAESOptimizer
            except ImportError:
                print("CMA-ES requires cma. Install with: pip install cma")
                sys.exit(1)

            agent_type = "SmartAgent" if args.smart else "ScorerAgent"
            opp_type = "Mixed (Lookahead2/Greedy/Defensive)" if args.mixed else "GreedyAgent"
            sigma = getattr(args, "sigma", 5.0)
            print(f"Running CMA-ES optimizer:")
            print(f"  Agent: {agent_type}")
            print(f"  Opponent: {opp_type}")
            print(f"  Max iterations: {args.generations}")
            print(f"  Games per eval: {args.games_per_eval}")
            print(f"  Sigma0: {sigma}")
            print()

            optimizer = CMAESOptimizer(
                games_per_eval=args.games_per_eval,
                num_workers=args.workers,
                maxiter=args.generations,
                sigma0=sigma,
                use_smart_agent=args.smart,
                use_mixed_opponents=args.mixed,
                use_lookahead=args.lookahead,
            )
            best_weights, best_fitness = optimizer.optimize()

        # Save intermediate weights for self-play rounds
        if self_play_rounds > 1:
            round_path = output_path.with_stem(f"{output_path.stem}_round{round_num}")
            with open(round_path, "w") as f:
                json.dump(best_weights.to_dict(), f, indent=2)
            print(f"Round {round_num} weights saved to: {round_path}")

            # Add best weights as custom opponent for next round
            if round_num < self_play_rounds:
                add_custom_opponent(best_weights)
                print(f"Added round {round_num} champion to opponent pool")

    # Save final weights
    weights_dict = best_weights.to_dict()
    with open(output_path, "w") as f:
        json.dump(weights_dict, f, indent=2)

    print(f"\nBest weights saved to: {output_path}")
    print(f"Best fitness: {best_fitness:.3f}")
    print(f"\nWeights:")
    for name, value in weights_dict.items():
        print(f"  {name}: {value:.3f}")


if __name__ == "__main__":
    main()
