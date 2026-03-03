#!/usr/bin/env python3
"""Analyze where MCTS disagrees with our linear scoring.

Finds positions where MCTS strongly prefers a move that our scoring ranks low,
then characterizes what makes those moves special (dual-purpose, tempo, forcing).
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.optimize_from_mcts import _collect_one_game
from sequence.scoring.scoring_function import SMART_WEIGHTS, FEATURE_NAMES


def analyze_disagreements(num_games: int = 50, mcts_iters: int = 1000):
    """Collect MCTS data and find where scoring disagrees."""
    weights = SMART_WEIGHTS.to_array()

    # Collect some games
    print(f"Collecting {num_games} games with {mcts_iters} MCTS iterations...")
    all_samples = []
    for seed in range(num_games):
        samples = _collect_one_game((seed, mcts_iters, seed % 2))
        all_samples.extend(samples)
        if (seed + 1) % 10 == 0:
            print(f"  {seed+1}/{num_games} ({len(all_samples)} positions)")

    print(f"\nTotal: {len(all_samples)} positions")

    # Analyze each position
    disagreements = []  # (position_idx, mcts_best, score_best, feature_diffs)
    dual_purpose_count = 0
    total_positions = 0

    for i, sample in enumerate(all_samples):
        features = sample["features"]  # (n_actions, 33)
        visits = sample["visits"]      # (n_actions,)

        if len(visits) < 2:
            continue

        total_positions += 1

        # MCTS ranking: best = most visits
        mcts_best = int(np.argmax(visits))
        mcts_visits_pct = visits[mcts_best] / visits.sum()

        # Our scoring ranking: best = highest dot(w, f)
        scores = features @ weights
        score_best = int(np.argmax(scores))

        # Disagreement?
        if mcts_best != score_best:
            # How bad is the disagreement?
            score_of_mcts_pick = scores[mcts_best]
            score_of_our_pick = scores[score_best]
            score_gap = score_of_our_pick - score_of_mcts_pick  # positive = we overvalue ours

            visits_of_our_pick = visits[score_best]
            visit_gap = visits[mcts_best] - visits_of_our_pick  # positive = MCTS strongly disagrees

            # Feature differences: what's special about MCTS's pick vs ours?
            f_mcts = features[mcts_best]
            f_ours = features[score_best]
            f_diff = f_mcts - f_ours

            disagreements.append({
                "idx": i,
                "mcts_confidence": mcts_visits_pct,
                "score_gap": score_gap,
                "visit_gap": visit_gap,
                "f_diff": f_diff,
                "f_mcts": f_mcts,
                "f_ours": f_ours,
            })

            # Check if MCTS pick is "dual-purpose": better on both offense AND defense
            # Offense features: four_in_a_row(1), three_in_a_row(2), two_in_a_row(3)
            # Defense features: opp_four_in_a_row(5), opp_three_in_a_row(6)
            offense_better = (f_diff[1] > 0 or f_diff[2] > 0 or f_diff[3] > 0)
            defense_better = (f_diff[5] < 0 or f_diff[6] < 0)  # less opponent = better
            if offense_better and defense_better:
                dual_purpose_count += 1

    agree_pct = 1 - len(disagreements) / total_positions
    print(f"\nAgreement rate: {agree_pct:.1%} ({total_positions - len(disagreements)}/{total_positions})")
    print(f"Disagreements: {len(disagreements)}")

    if not disagreements:
        return

    # Sort by MCTS confidence (strongest disagreements first)
    disagreements.sort(key=lambda d: d["mcts_confidence"], reverse=True)

    # Aggregate feature differences across all disagreements
    print(f"\n{'='*60}")
    print("WHAT MCTS VALUES MORE THAN OUR SCORING")
    print(f"{'='*60}")
    print("(Average feature difference: MCTS_pick - OUR_pick)")
    print("Positive = MCTS pick has MORE of this feature\n")

    all_diffs = np.array([d["f_diff"] for d in disagreements])
    mean_diffs = all_diffs.mean(axis=0)

    # Sort by absolute difference
    sorted_features = sorted(
        enumerate(FEATURE_NAMES), key=lambda x: abs(mean_diffs[x[0]]), reverse=True
    )

    for idx, name in sorted_features:
        diff = mean_diffs[idx]
        if abs(diff) > 0.01:
            direction = "MORE" if diff > 0 else "LESS"
            print(f"  {name:30s}: {diff:+.3f} ({direction})")

    # Dual-purpose analysis
    print(f"\n{'='*60}")
    print("DUAL-PURPOSE ANALYSIS")
    print(f"{'='*60}")
    print(f"Positions where MCTS picks a move that's better on BOTH offense AND defense:")
    print(f"  {dual_purpose_count}/{len(disagreements)} disagreements ({dual_purpose_count/max(1,len(disagreements)):.0%})")

    # Interaction analysis: look for positions where MCTS pick has
    # BOTH high offensive gain AND high defensive gain
    print(f"\n{'='*60}")
    print("INTERACTION PATTERNS IN DISAGREEMENTS")
    print(f"{'='*60}")

    # Count how many positive effects MCTS's pick has vs ours
    effect_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    for d in disagreements:
        f_diff = d["f_diff"]
        positive_effects = 0
        # Offensive improvement (own line advancement)
        if f_diff[1] > 0 or f_diff[2] > 0:  # four/three in a row
            positive_effects += 1
        # Defensive improvement (reduce opponent lines)
        if f_diff[5] < 0 or f_diff[6] < 0:  # opp four/three
            positive_effects += 1
        # Positional improvement (fork, clustering, shared lines)
        if f_diff[22] > 0:  # fork_count
            positive_effects += 1
        # Card efficiency (hand improvement)
        if f_diff[14] < 0:  # fewer dead cards
            positive_effects += 1
        effect_counts[min(positive_effects, 4)] += 1

    print("\nNumber of simultaneous positive effects in MCTS's preferred move:")
    for n, count in sorted(effect_counts.items()):
        bar = "#" * (count * 40 // max(1, len(disagreements)))
        print(f"  {n} effects: {count:3d} ({count/max(1,len(disagreements)):5.1%}) {bar}")

    # Strong disagreements (MCTS very confident, our score very different)
    strong = [d for d in disagreements if d["mcts_confidence"] > 0.5 and d["score_gap"] > 5]
    print(f"\n{'='*60}")
    print(f"STRONG DISAGREEMENTS (MCTS >50% visits, score gap >5)")
    print(f"{'='*60}")
    print(f"Count: {len(strong)}")

    if strong:
        print("\nTop 5 feature patterns in strong disagreements:")
        strong_diffs = np.array([d["f_diff"] for d in strong])
        strong_mean = strong_diffs.mean(axis=0)
        sorted_strong = sorted(
            enumerate(FEATURE_NAMES), key=lambda x: abs(strong_mean[x[0]]), reverse=True
        )
        for idx, name in sorted_strong[:10]:
            diff = strong_mean[idx]
            if abs(diff) > 0.01:
                print(f"  {name:30s}: {diff:+.3f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=50)
    parser.add_argument("--mcts-iters", type=int, default=1000)
    args = parser.parse_args()
    analyze_disagreements(args.games, args.mcts_iters)
