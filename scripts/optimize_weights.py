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
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    from sequence.scoring.optimizer import GeneticOptimizer

    if args.method == "ga":
        print(f"Running Genetic Algorithm:")
        print(f"  Generations: {args.generations}")
        print(f"  Population: {args.population}")
        print(f"  Games per eval: {args.games_per_eval}")
        print()

        optimizer = GeneticOptimizer(
            population_size=args.population,
            num_generations=args.generations,
            games_per_eval=args.games_per_eval,
            num_workers=args.workers,
        )
        best_weights, best_fitness = optimizer.optimize()

    elif args.method == "cmaes":
        try:
            from sequence.scoring.optimizer import CMAESOptimizer
        except ImportError:
            print("CMA-ES requires scipy. Install with: pip install scipy")
            sys.exit(1)

        print("Running CMA-ES optimizer...")
        optimizer = CMAESOptimizer(
            games_per_eval=args.games_per_eval,
            num_workers=args.workers,
        )
        best_weights = optimizer.optimize()
        best_fitness = -1.0  # Not tracked in CMA-ES

    # Save
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
