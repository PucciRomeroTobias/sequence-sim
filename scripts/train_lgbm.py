#!/usr/bin/env python3
"""Train a LightGBM LambdaRank model for Sequence action ranking.

Usage:
    # Train from existing MCTS data
    python scripts/train_lgbm.py --data data/nn/training_data_combined.npz \
        --output data/lgbm/model.txt

    # With validation tournament
    python scripts/train_lgbm.py --data data/nn/training_data_combined.npz \
        --output data/lgbm/model.txt --validate --validation-games 200
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def validate_tournament(model_path: str, num_games: int) -> dict[str, float]:
    """Run tournament: LGBMAgent vs SmartAgent and others."""
    import random

    from sequence.agents.lgbm_agent import LGBMAgent
    from sequence.agents.smart_agent import SmartAgent
    from sequence.agents.greedy_agent import GreedyAgent
    from sequence.agents.defensive_agent import DefensiveAgent
    from sequence.core.board import Board
    from sequence.core.deck import Deck
    from sequence.core.game import GameConfig
    from sequence.core.game_state import GameState
    from sequence.core.types import TeamId

    def play_games(make_agent, make_opponent, n_games):
        wins = 0
        for seed in range(n_games):
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

            our_team_val = seed % 2
            our_team = TeamId(our_team_val)
            opp_team = TeamId(1 - our_team_val)

            state = GameState(
                board=board, hands=hands, deck=deck,
                current_team=TeamId.TEAM_0, num_teams=2, sequences_to_win=2,
            )

            agent = make_agent()
            opp = make_opponent()

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

    results = {}

    # LGBMAgent vs SmartAgent
    print("  LGBM vs SmartAgent...", end=" ", flush=True)
    t0 = time.time()
    wr = play_games(
        lambda: LGBMAgent(model_path),
        lambda: SmartAgent(use_lookahead=True),
        num_games,
    )
    print(f"{wr:.1%} ({time.time()-t0:.1f}s)")
    results["vs_smart"] = wr

    # SmartAgent vs SmartAgent (baseline)
    print("  SmartAgent vs SmartAgent...", end=" ", flush=True)
    t0 = time.time()
    wr = play_games(
        lambda: SmartAgent(use_lookahead=True),
        lambda: SmartAgent(use_lookahead=True),
        num_games,
    )
    print(f"{wr:.1%} ({time.time()-t0:.1f}s)")
    results["smart_vs_smart"] = wr

    # LGBMAgent vs Greedy
    print("  LGBM vs Greedy...", end=" ", flush=True)
    t0 = time.time()
    wr = play_games(
        lambda: LGBMAgent(model_path),
        lambda: GreedyAgent(),
        num_games,
    )
    print(f"{wr:.1%} ({time.time()-t0:.1f}s)")
    results["vs_greedy"] = wr

    # SmartAgent vs Greedy (baseline)
    print("  SmartAgent vs Greedy...", end=" ", flush=True)
    t0 = time.time()
    wr = play_games(
        lambda: SmartAgent(use_lookahead=True),
        lambda: GreedyAgent(),
        num_games,
    )
    print(f"{wr:.1%} ({time.time()-t0:.1f}s)")
    results["smart_vs_greedy"] = wr

    # LGBMAgent vs Defensive
    print("  LGBM vs Defensive...", end=" ", flush=True)
    t0 = time.time()
    wr = play_games(
        lambda: LGBMAgent(model_path),
        lambda: DefensiveAgent(),
        num_games,
    )
    print(f"{wr:.1%} ({time.time()-t0:.1f}s)")
    results["vs_defensive"] = wr

    # SmartAgent vs Defensive (baseline)
    print("  SmartAgent vs Defensive...", end=" ", flush=True)
    t0 = time.time()
    wr = play_games(
        lambda: SmartAgent(use_lookahead=True),
        lambda: DefensiveAgent(),
        num_games,
    )
    print(f"{wr:.1%} ({time.time()-t0:.1f}s)")
    results["smart_vs_defensive"] = wr

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Train LightGBM LambdaRank for Sequence"
    )
    parser.add_argument(
        "--data", type=str, required=True,
        help="Path to MCTS training data (.npz)",
    )
    parser.add_argument(
        "--output", type=str, default="data/lgbm/model.txt",
        help="Path to save trained model",
    )
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--min-child-samples", type=int, default=20)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--validation-games", type=int, default=200)
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Train ---
    print("=" * 60)
    print("Training LightGBM LambdaRank")
    print("=" * 60)
    print(f"  Data: {args.data}")
    print(f"  Leaves: {args.num_leaves}, Estimators: {args.n_estimators}")
    print(f"  LR: {args.lr}, Min child: {args.min_child_samples}")
    print()

    from sequence.scoring.lgbm_scoring import train_lgbm_ranker

    t0 = time.time()
    model = train_lgbm_ranker(
        data_path=args.data,
        num_leaves=args.num_leaves,
        n_estimators=args.n_estimators,
        learning_rate=args.lr,
        min_child_samples=args.min_child_samples,
    )
    t1 = time.time()
    print(f"\nTraining done ({t1-t0:.1f}s)")

    # Save
    model.save_model(str(output_path))
    print(f"Model saved to {output_path}")

    # --- Validate ---
    if args.validate:
        print()
        print("=" * 60)
        print("Validation tournament")
        print("=" * 60)
        print(f"  Games per matchup: {args.validation_games}")
        print()

        t2 = time.time()
        results = validate_tournament(str(output_path), args.validation_games)
        t3 = time.time()

        print()
        print("=" * 60)
        print("RESULTS")
        print("=" * 60)
        print(f"  {'Matchup':<30} {'LGBM':>8} {'Smart':>8}")
        print(f"  {'vs SmartAgent':<30} {results['vs_smart']:>7.1%} {results.get('smart_vs_smart', 0):>7.1%}")
        print(f"  {'vs Greedy':<30} {results['vs_greedy']:>7.1%} {results['smart_vs_greedy']:>7.1%}")
        print(f"  {'vs Defensive':<30} {results['vs_defensive']:>7.1%} {results['smart_vs_defensive']:>7.1%}")
        print(f"\n  Validation done ({t3-t2:.1f}s)")


if __name__ == "__main__":
    main()
