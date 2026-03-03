#!/usr/bin/env python3
"""Train a neural network for Sequence state evaluation.

End-to-end pipeline:
1. Generate MCTS oracle dataset (slow, parallelizable)
2. Train MLP on pairwise ranking loss (fast, CPU)
3. Validate via tournament against mixed opponents (optional)

Usage:
    # Generate data only
    python scripts/train_neural.py --generate 500 --mcts-iters 1000 --workers 9 \
        --save-data data/nn/training_data.npz

    # Train from existing data
    python scripts/train_neural.py --load-data data/nn/training_data.npz \
        --hidden 128 --epochs 100 --lr 0.001 --batch-size 256 \
        --output data/nn/model.pt

    # All-in-one
    python scripts/train_neural.py --generate 500 --mcts-iters 1000 --workers 9 \
        --save-data data/nn/training_data.npz \
        --output data/nn/model.pt

    # With validation tournament
    python scripts/train_neural.py --load-data data/nn/training_data.npz \
        --output data/nn/model.pt --validate --validation-games 200
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def save_dataset(samples: list[dict], path: str) -> None:
    """Save dataset to npz format compatible with train_model()."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lengths = []
    feature_arrays = {}
    visit_arrays = {}

    for i, sample in enumerate(samples):
        feature_arrays[f"f_{i}"] = sample["features"]
        visit_arrays[f"v_{i}"] = sample["visits"]
        lengths.append(len(sample["visits"]))

    np.savez(
        out_path,
        lengths=np.array(lengths),
        **feature_arrays,
        **visit_arrays,
    )
    print(f"Dataset saved to {out_path} ({len(samples)} positions)")


def load_dataset(path: str) -> list[dict]:
    """Load dataset from npz format."""
    data = np.load(path, allow_pickle=True)
    lengths = data["lengths"]
    samples: list[dict] = []
    for j in range(len(lengths)):
        samples.append({
            "features": data[f"f_{j}"],
            "visits": data[f"v_{j}"],
        })
    return samples


def compute_linear_baseline(samples: list[dict]) -> float:
    """Compute pairwise accuracy of linear SMART_WEIGHTS as baseline."""
    from sequence.scoring.scoring_function import SMART_WEIGHTS

    weights = SMART_WEIGHTS.to_array()
    correct = 0
    total = 0

    for sample in samples:
        features = sample["features"]
        visits = sample["visits"]
        best_idx = int(np.argmax(visits))
        f_best = features[best_idx]

        sorted_indices = np.argsort(-visits)
        for idx in sorted_indices[1:6]:
            if visits[idx] < visits[best_idx]:
                diff = f_best - features[idx]
                if np.dot(weights, diff) > 0:
                    correct += 1
                total += 1

    return correct / max(total, 1)


def validate_tournament(
    model_path: str, num_games: int, hidden: int = 128,
) -> dict[str, float]:
    """Run tournament: NeuralAgent vs mixed opponents, compared to SmartAgent."""
    from sequence.agents.neural_agent import NeuralAgent
    from sequence.agents.smart_agent import SmartAgent
    from sequence.agents.greedy_agent import GreedyAgent
    from sequence.agents.defensive_agent import DefensiveAgent
    from sequence.core.game import GameConfig
    from sequence.core.game_state import GameState
    from sequence.core.board import Board
    from sequence.core.deck import Deck
    from sequence.core.types import TeamId

    import random

    opponent_classes = [GreedyAgent, DefensiveAgent, SmartAgent]

    def play_games(make_agent, n_games):
        wins = 0
        for seed in range(n_games):
            rng = random.Random(seed)
            config = GameConfig(seed=seed, max_turns=200)
            board = Board()
            deck = Deck(seed=seed)
            hands: dict[int, list] = {}
            for t in range(2):
                hand = []
                for _ in range(7):
                    c = deck.draw()
                    if c is not None:
                        hand.append(c)
                hands[t] = hand

            # Alternate sides
            our_team_val = seed % 2
            our_team = TeamId(our_team_val)
            opp_team = TeamId(1 - our_team_val)

            state = GameState(
                board=board, hands=hands, deck=deck,
                current_team=TeamId.TEAM_0, num_teams=2, sequences_to_win=2,
            )

            agent = make_agent()
            opp_cls = rng.choice(opponent_classes)
            opp = opp_cls() if opp_cls != SmartAgent else SmartAgent(use_lookahead=False)

            agents = [None, None]
            agents[our_team_val] = agent
            agents[1 - our_team_val] = opp

            agent.notify_game_start(our_team, config)
            opp.notify_game_start(opp_team, config)

            for _turn in range(config.max_turns):
                winner = state.is_terminal()
                if winner is not None:
                    if winner == our_team:
                        wins += 1
                    break

                team = state.current_team
                legal_actions = state.get_legal_actions(team)
                if not legal_actions:
                    break

                current_agent = agents[team.value]
                action = current_agent.choose_action(state, legal_actions)

                for a in agents:
                    a.notify_action(action, team)
                state = state.apply_action(action)

        return wins / max(n_games, 1)

    neural_wr = play_games(
        lambda: NeuralAgent(model_path, use_lookahead=True, hidden=hidden),
        num_games,
    )
    smart_wr = play_games(
        lambda: SmartAgent(use_lookahead=True),
        num_games,
    )

    return {"neural": neural_wr, "smart": smart_wr}


def main():
    parser = argparse.ArgumentParser(
        description="Train neural network for Sequence state evaluation"
    )
    # Data generation
    parser.add_argument(
        "--generate", type=int, default=0,
        help="Number of MCTS games to generate (0 = skip generation)",
    )
    parser.add_argument(
        "--mcts-iters", type=int, default=1000,
        help="MCTS iterations per move (default: 1000)",
    )
    parser.add_argument(
        "--workers", type=int, default=9,
        help="Parallel workers for data generation (default: 9)",
    )
    parser.add_argument(
        "--save-data", type=str, default="",
        help="Path to save generated dataset (.npz)",
    )
    parser.add_argument(
        "--load-data", type=str, default="",
        help="Path to load existing dataset (.npz)",
    )
    # Training
    parser.add_argument("--hidden", type=int, default=128, help="Hidden layer size")
    parser.add_argument("--epochs", type=int, default=100, help="Max training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument(
        "--output", type=str, default="",
        help="Path to save trained model (.pt)",
    )
    # Validation
    parser.add_argument(
        "--validate", action="store_true",
        help="Run validation tournament after training",
    )
    parser.add_argument(
        "--validation-games", type=int, default=200,
        help="Games per agent in validation tournament",
    )
    args = parser.parse_args()

    if not args.generate and not args.load_data:
        parser.error("Must specify --generate N or --load-data PATH")

    samples: list[dict] = []

    # --- Step 1: Generate or load data ---
    if args.generate > 0:
        print("=" * 60)
        print("STEP 1: Generating MCTS oracle dataset (informed determinization)")
        print("=" * 60)
        print(f"  Games: {args.generate}")
        print(f"  MCTS iterations: {args.mcts_iters}")
        print(f"  Workers: {args.workers}")
        if args.save_data:
            print(f"  Incremental save to: {args.save_data}")
        print()

        from scripts.optimize_from_mcts import _collect_one_game

        # Load existing data if resuming
        save_path = args.save_data
        if save_path and Path(save_path).exists():
            samples = load_dataset(save_path)
            start_game = len(set())  # We don't track game IDs, count positions
            print(f"  Resuming: loaded {len(samples)} existing positions")
        else:
            samples = []

        t0 = time.time()
        try:
            for seed in range(args.generate):
                game_samples = _collect_one_game((seed, args.mcts_iters, seed % 2))
                samples.extend(game_samples)

                elapsed = time.time() - t0
                avg = elapsed / (seed + 1)
                remaining = avg * (args.generate - seed - 1)
                print(
                    f"  Game {seed+1}/{args.generate}: "
                    f"+{len(game_samples)} positions "
                    f"(total: {len(samples)}) "
                    f"[{elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining]"
                )

                # Save every 10 games
                if save_path and (seed + 1) % 10 == 0:
                    save_dataset(samples, save_path)
        except KeyboardInterrupt:
            print(f"\n\nInterrupted after {seed + 1} games, {len(samples)} positions")

        t1 = time.time()
        print(f"\nDataset: {len(samples)} positions from {args.generate} games ({t1-t0:.1f}s)")

        if save_path:
            save_dataset(samples, save_path)

    if args.load_data:
        print(f"\nLoading dataset from {args.load_data}")
        samples = load_dataset(args.load_data)
        print(f"Loaded {len(samples)} positions")

    if not samples:
        print("ERROR: No training data available!")
        sys.exit(1)

    # --- Linear baseline ---
    print()
    baseline_acc = compute_linear_baseline(samples)
    print(f"Linear baseline (SMART_WEIGHTS) pairwise accuracy: {baseline_acc:.1%}")

    # --- Step 2: Train ---
    if not args.output:
        print("\nNo --output specified, skipping training.")
        return

    print()
    print("=" * 60)
    print("STEP 2: Training neural network")
    print("=" * 60)
    print(f"  Hidden: {args.hidden}")
    print(f"  Epochs: {args.epochs}")
    print(f"  LR: {args.lr}")
    print(f"  Batch size: {args.batch_size}")
    print()

    # Save temp data for train_model if we generated it
    import tempfile
    if args.load_data:
        data_path = args.load_data
    else:
        tmp = tempfile.NamedTemporaryFile(suffix=".npz", delete=False)
        data_path = tmp.name
        tmp.close()
        save_dataset(samples, data_path)

    from sequence.scoring.neural_scoring import train_model
    import torch

    t2 = time.time()
    net = train_model(
        data_path=data_path,
        hidden=args.hidden,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
    )
    t3 = time.time()
    print(f"\nTraining done ({t3-t2:.1f}s)")

    # Save model
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), output_path)
    print(f"Model saved to {output_path}")

    # Parameter count
    total_params = sum(p.numel() for p in net.parameters())
    print(f"Parameters: {total_params:,}")

    # Cleanup temp file
    if not args.load_data:
        import os
        os.unlink(data_path)

    # --- Step 3: Validate ---
    if args.validate:
        print()
        print("=" * 60)
        print("STEP 3: Validation tournament")
        print("=" * 60)
        print(f"  Games: {args.validation_games}")
        print()

        t4 = time.time()
        results = validate_tournament(
            str(output_path), args.validation_games, hidden=args.hidden,
        )
        t5 = time.time()

        print(f"\n  Neural vs mixed: {results['neural']:.1%}")
        print(f"  Smart vs mixed:  {results['smart']:.1%}")
        print(f"  Improvement:     {results['neural'] - results['smart']:+.1%}")
        print(f"  Validation done ({t5-t4:.1f}s)")

    # --- Summary ---
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Dataset: {len(samples)} positions")
    print(f"  Linear baseline accuracy: {baseline_acc:.1%}")
    print(f"  Model: {output_path}")
    print(f"  Parameters: {total_params:,}")


if __name__ == "__main__":
    main()
