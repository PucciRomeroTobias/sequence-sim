#!/usr/bin/env python3
"""Optimize scoring weights using MCTS oracle ranking.

Instead of playing hundreds of noisy games to evaluate each weight candidate,
this script:
1. Generates a dataset of board positions with MCTS oracle rankings
2. Optimizes weights to match MCTS rankings using pairwise ranking loss
3. Validates the result with a tournament

The key insight: score = dot(weights, features) is linear, so matching MCTS
rankings becomes a pairwise classification problem that's fast and noise-free.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# Step 1: Generate MCTS oracle dataset
# ---------------------------------------------------------------------------


def _collect_one_game(args: tuple[int, int, int]) -> list[dict]:
    """Play one MCTS game and collect training samples.

    Each sample is a (features_matrix, visits_vector) pair for one position.
    Returns list of serializable dicts with numpy arrays.
    """
    seed, mcts_iters, mcts_team = args

    from sequence.agents.mcts_agent import MCTSAgent
    from sequence.agents.smart_agent import SmartAgent
    from sequence.core.board import Board
    from sequence.core.card_tracker import CardTracker
    from sequence.core.deck import Deck
    from sequence.core.game import GameConfig
    from sequence.core.game_state import GameState
    from sequence.core.types import TeamId
    from sequence.scoring.features import extract_features

    config = GameConfig(seed=seed, max_turns=200)
    board = Board()
    deck = Deck(seed=config.seed)
    hands: dict[int, list] = {}
    for t in range(2):
        hand = []
        for _ in range(7):
            c = deck.draw()
            if c is not None:
                hand.append(c)
        hands[t] = hand

    state = GameState(
        board=board, hands=hands, deck=deck,
        current_team=TeamId.TEAM_0, num_teams=2, sequences_to_win=2,
    )

    mcts = MCTSAgent(
        iterations=mcts_iters,
        use_heuristic_rollout=True,
        use_informed_determinization=True,
        seed=seed,
    )
    smart = SmartAgent(use_lookahead=False)

    agents = [None, None]
    agents[mcts_team] = mcts
    agents[1 - mcts_team] = smart

    mcts_tid = TeamId(mcts_team)
    other_tid = TeamId(1 - mcts_team)

    mcts.notify_game_start(mcts_tid, config)
    smart.notify_game_start(other_tid, config)

    tracker = CardTracker(mcts_tid, 2)

    samples: list[dict] = []

    for _turn in range(config.max_turns):
        winner = state.is_terminal()
        if winner is not None:
            break

        team = state.current_team
        legal_actions = state.get_legal_actions(team)
        if not legal_actions:
            break

        agent = agents[team.value]
        action = agent.choose_action(state, legal_actions)

        # Collect training data from MCTS turns with multiple choices
        if team.value == mcts_team and len(legal_actions) > 1 and mcts.last_mcts_visits:
            from sequence.core.actions import ActionType
            from sequence.core.types import EMPTY as EMPTY_VAL

            features_list = []
            visits_list = []
            board = state.board
            chips = board.chips
            hand = state.hands.get(mcts_team, [])

            for a in legal_actions:
                # Virtual apply for PLACE actions (fast path)
                if a.action_type == ActionType.PLACE and a.position is not None:
                    pos = a.position
                    r, c = pos.row, pos.col
                    old_chip = int(chips[r, c])
                    chips[r, c] = mcts_team
                    board.empty_positions.discard(pos)
                    hand.remove(a.card)
                    state.turn_number += 1

                    feats = extract_features(state, mcts_tid, tracker=tracker)

                    state.turn_number -= 1
                    hand.append(a.card)
                    chips[r, c] = old_chip
                    if old_chip == EMPTY_VAL:
                        board.empty_positions.add(pos)
                else:
                    next_state = state.apply_action(a)
                    feats = extract_features(next_state, mcts_tid, tracker=tracker)

                features_list.append(feats)
                visits_list.append(mcts.last_mcts_visits.get(str(a), 0))

            total_visits = sum(visits_list)
            if total_visits > 0 and max(visits_list) > total_visits * 0.1:
                samples.append({
                    "features": np.array(features_list),
                    "visits": np.array(visits_list, dtype=np.float64),
                })

        for a in agents:
            a.notify_action(action, team)
        tracker.on_action(action, team)
        state = state.apply_action(action)

    return samples


def _run_worker_batch(seeds: list[int], mcts_iters: int, out_path: str) -> None:
    """Worker entry point: collect games for given seeds, save to numpy file."""
    all_features = []
    all_visits = []
    all_lengths = []

    for seed in seeds:
        samples = _collect_one_game((seed, mcts_iters, seed % 2))
        for s in samples:
            all_features.append(s["features"])
            all_visits.append(s["visits"])
            all_lengths.append(len(s["visits"]))

    np.savez(
        out_path,
        lengths=np.array(all_lengths),
        **{f"f_{i}": f for i, f in enumerate(all_features)},
        **{f"v_{i}": v for i, v in enumerate(all_visits)},
    )


def generate_dataset(
    num_games: int, mcts_iters: int, num_workers: int,
) -> list[dict]:
    """Generate training dataset using subprocess-based parallelism."""

    if num_workers <= 1:
        all_samples: list[dict] = []
        for seed in range(num_games):
            samples = _collect_one_game((seed, mcts_iters, seed % 2))
            all_samples.extend(samples)
            if (seed + 1) % 10 == 0 or seed + 1 == num_games:
                print(f"  Games: {seed+1}/{num_games} ({len(all_samples)} positions)")
        return all_samples

    # Split seeds into batches for workers
    all_seeds = list(range(num_games))
    batch_size = max(1, len(all_seeds) // num_workers)
    batches = [
        all_seeds[i:i + batch_size]
        for i in range(0, len(all_seeds), batch_size)
    ]

    # Launch each batch as a subprocess
    script_path = Path(__file__).resolve()
    tmp_dir = tempfile.mkdtemp(prefix="mcts_data_")
    procs: list[tuple[subprocess.Popen, str]] = []

    for i, batch in enumerate(batches):
        out_path = os.path.join(tmp_dir, f"batch_{i}.npz")
        seeds_str = ",".join(str(s) for s in batch)
        cmd = [
            sys.executable, str(script_path),
            "--_worker",
            "--_seeds", seeds_str,
            "--_mcts-iters", str(mcts_iters),
            "--_out", out_path,
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        procs.append((proc, out_path))

    print(f"  Launched {len(procs)} worker processes ({batch_size} games each)")

    # Wait for all workers
    all_samples = []
    for i, (proc, out_path) in enumerate(procs):
        proc.wait()
        if proc.returncode != 0:
            stderr = proc.stderr.read().decode() if proc.stderr else ""
            print(f"  WARNING: Worker {i} failed: {stderr[:200]}")
            continue
        try:
            data = np.load(out_path, allow_pickle=True)
            lengths = data["lengths"]
            for j in range(len(lengths)):
                all_samples.append({
                    "features": data[f"f_{j}"],
                    "visits": data[f"v_{j}"],
                })
        except Exception as e:
            print(f"  WARNING: Failed to load batch {i}: {e}")
        print(f"  Worker {i+1}/{len(procs)} done ({len(all_samples)} positions so far)")

    # Cleanup
    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)

    return all_samples


# ---------------------------------------------------------------------------
# Step 2: Pairwise ranking optimization
# ---------------------------------------------------------------------------


def build_pairwise_data(
    samples: list[dict], top_k_pairs: int = 5,
) -> np.ndarray:
    """Build pairwise feature differences from MCTS oracle data.

    For each position, creates pairs (best_action, other_action) where
    best = highest MCTS visits. We want dot(w, f_best - f_other) > 0.

    Returns (N_pairs, N_features) array of feature differences.
    """
    diffs: list[np.ndarray] = []

    for sample in samples:
        features = sample["features"]  # (n_actions, n_features)
        visits = sample["visits"]      # (n_actions,)

        best_idx = int(np.argmax(visits))
        f_best = features[best_idx]

        # Create pairs with top_k other actions (sorted by visits descending)
        sorted_indices = np.argsort(-visits)
        for idx in sorted_indices[1:top_k_pairs + 1]:
            if visits[idx] < visits[best_idx]:
                diff = f_best - features[idx]
                diffs.append(diff)

    if not diffs:
        raise ValueError("No pairwise data generated")

    return np.array(diffs)


def optimize_weights(
    diffs: np.ndarray,
    initial_weights: np.ndarray,
    l2_lambda: float = 0.01,
) -> np.ndarray:
    """Optimize weights using pairwise logistic ranking loss.

    Loss = sum log(1 + exp(-dot(w, diff_i))) + l2_lambda * weighted_L2

    The L2 penalty is proportional to the magnitude of each initial weight,
    so high-impact weights (completed_sequences=91, opp_completed=-100) are
    anchored more strongly while low-impact ones are free to move.

    This is convex and has a unique global minimum.
    """
    from scipy.optimize import minimize

    n_pairs = diffs.shape[0]

    # Per-weight regularization: anchor strong weights harder
    weight_importance = np.maximum(np.abs(initial_weights), 1.0)
    reg_scale = l2_lambda * weight_importance

    def loss_and_grad(w: np.ndarray) -> tuple[float, np.ndarray]:
        scores = diffs @ w  # (n_pairs,)
        # Numerically stable logistic loss
        neg_scores = -scores
        max_val = np.maximum(neg_scores, 0)
        log_sum = max_val + np.log(np.exp(-max_val) + np.exp(neg_scores - max_val))
        # Weighted L2 regularization toward initial weights
        delta = w - initial_weights
        loss = np.mean(log_sum) + np.sum(reg_scale * delta ** 2)

        # Gradient
        sigmoid_neg = 1.0 / (1.0 + np.exp(np.clip(scores, -500, 500)))
        grad = -diffs.T @ sigmoid_neg / n_pairs + 2 * reg_scale * delta

        return float(loss), grad

    result = minimize(
        loss_and_grad,
        initial_weights,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 500},
    )

    return result.x


# ---------------------------------------------------------------------------
# Step 3: Validation tournament
# ---------------------------------------------------------------------------


def run_validation(
    weights: np.ndarray, num_games: int, num_workers: int,
) -> dict[str, float]:
    """Run a quick tournament to validate the optimized weights."""
    from sequence.scoring.optimizer import _evaluate_weights_smart

    # vs mixed opponents
    args = (weights, num_games, True, False)  # (weights, games, mixed, lookahead)
    win_rate_mixed = _evaluate_weights_smart(args)

    return {"vs_mixed": win_rate_mixed}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Optimize weights using MCTS oracle ranking"
    )
    parser.add_argument(
        "--games", type=int, default=200,
        help="Number of MCTS games to generate (default: 200)",
    )
    parser.add_argument(
        "--mcts-iters", type=int, default=1000,
        help="MCTS iterations per move (default: 1000)",
    )
    parser.add_argument(
        "--workers", type=int, default=9,
        help="Parallel workers (default: 9)",
    )
    parser.add_argument(
        "--l2", type=float, default=0.001,
        help="L2 regularization strength (default: 0.001)",
    )
    parser.add_argument(
        "--output", type=str, default="data/weights/optimized_mcts.json",
        help="Output file for optimized weights",
    )
    parser.add_argument(
        "--validation-games", type=int, default=200,
        help="Games for validation tournament (default: 200)",
    )
    parser.add_argument(
        "--top-k-pairs", type=int, default=5,
        help="Number of pairwise comparisons per position (default: 5)",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    from sequence.scoring.scoring_function import SMART_WEIGHTS

    # --- Step 1: Generate dataset ---
    print("=" * 60)
    print("STEP 1: Generating MCTS oracle dataset")
    print("=" * 60)
    print(f"  Games: {args.games}")
    print(f"  MCTS iterations: {args.mcts_iters}")
    print(f"  Workers: {args.workers}")
    print()

    t0 = time.time()
    samples = generate_dataset(args.games, args.mcts_iters, args.workers)
    t1 = time.time()
    print(f"\nDataset: {len(samples)} positions from {args.games} games ({t1-t0:.1f}s)")

    if not samples:
        print("ERROR: No training data collected!")
        sys.exit(1)

    # --- Step 2: Optimize ---
    print()
    print("=" * 60)
    print("STEP 2: Optimizing weights (pairwise ranking loss)")
    print("=" * 60)

    diffs = build_pairwise_data(samples, top_k_pairs=args.top_k_pairs)
    print(f"  Pairwise samples: {diffs.shape[0]}")
    print(f"  Features: {diffs.shape[1]}")
    print(f"  L2 lambda: {args.l2}")
    print()

    initial = SMART_WEIGHTS.to_array()
    t2 = time.time()
    optimized_arr = optimize_weights(diffs, initial, l2_lambda=args.l2)
    t3 = time.time()
    print(f"\nOptimization done ({t3-t2:.1f}s)")

    from sequence.scoring.scoring_function import ScoringWeights, FEATURE_NAMES

    best_weights = ScoringWeights.from_array(optimized_arr)

    # Show weight changes
    print("\nWeight changes (SMART_WEIGHTS -> optimized):")
    initial_dict = SMART_WEIGHTS.to_dict()
    optimized_dict = best_weights.to_dict()
    for name in FEATURE_NAMES:
        old = initial_dict[name]
        new = optimized_dict[name]
        if abs(new - old) > 0.1:
            print(f"  {name}: {old:.2f} -> {new:.2f} ({new-old:+.2f})")

    # Pairwise accuracy on training data
    train_scores = diffs @ optimized_arr
    accuracy = np.mean(train_scores > 0)
    print(f"\nTraining pairwise accuracy: {accuracy:.1%}")
    baseline_scores = diffs @ initial
    baseline_acc = np.mean(baseline_scores > 0)
    print(f"Baseline (SMART_WEIGHTS) accuracy: {baseline_acc:.1%}")

    # --- Step 3: Save ---
    weights_dict = best_weights.to_dict()
    with open(output_path, "w") as f:
        json.dump(weights_dict, f, indent=2)
    print(f"\nWeights saved to: {output_path}")

    # --- Step 4: Validate ---
    print()
    print("=" * 60)
    print("STEP 3: Validation tournament")
    print("=" * 60)
    print(f"  Games: {args.validation_games}")
    print()

    t4 = time.time()
    results = run_validation(optimized_arr, args.validation_games, args.workers)
    t5 = time.time()
    print(f"  Win rate vs mixed opponents: {results['vs_mixed']:.1%}")

    # Compare with SMART_WEIGHTS baseline
    baseline = run_validation(initial, args.validation_games, args.workers)
    print(f"  Baseline (SMART_WEIGHTS):    {baseline['vs_mixed']:.1%}")
    print(f"  Improvement:                 {results['vs_mixed'] - baseline['vs_mixed']:+.1%}")
    print(f"\nValidation done ({t5-t4:.1f}s)")

    # --- Summary ---
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Dataset: {len(samples)} positions, {diffs.shape[0]} pairs")
    print(f"  Pairwise accuracy: {baseline_acc:.1%} -> {accuracy:.1%}")
    print(f"  Win rate vs mixed: {baseline['vs_mixed']:.1%} -> {results['vs_mixed']:.1%}")
    print(f"  Total time: {t5-t0:.0f}s")


if __name__ == "__main__":
    # Worker subprocess mode
    if "--_worker" in sys.argv:
        parser = argparse.ArgumentParser()
        parser.add_argument("--_worker", action="store_true")
        parser.add_argument("--_seeds", type=str, required=True)
        parser.add_argument("--_mcts-iters", type=int, required=True)
        parser.add_argument("--_out", type=str, required=True)
        args = parser.parse_args()
        seeds = [int(s) for s in args._seeds.split(",")]
        _run_worker_batch(seeds, getattr(args, "_mcts_iters"), args._out)
        sys.exit(0)

    main()
