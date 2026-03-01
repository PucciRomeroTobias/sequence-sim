#!/usr/bin/env python3
"""Generate human-readable strategy tips from optimized weights."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    parser = argparse.ArgumentParser(description="Explain scoring strategy")
    parser.add_argument(
        "--weights",
        type=str,
        default="data/weights/optimized.json",
        help="Path to weights JSON file",
    )
    args = parser.parse_args()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        print(f"Weights file not found: {weights_path}")
        print("Run optimize_weights.py first, or use --weights to specify a file.")
        sys.exit(1)

    from sequence.scoring.scoring_function import ScoringWeights
    from sequence.analysis.explainer import explain_weights, generate_report

    with open(weights_path) as f:
        data = json.load(f)

    weights = ScoringWeights.from_dict(data)

    print("=" * 60)
    print("  SEQUENCE STRATEGY ANALYSIS")
    print("=" * 60)
    print()

    tips = explain_weights(weights)
    for i, tip in enumerate(tips, 1):
        print(f"  {i}. {tip}")

    print()
    print("=" * 60)
    print("  FULL REPORT")
    print("=" * 60)
    print()
    print(generate_report(weights))


if __name__ == "__main__":
    main()
