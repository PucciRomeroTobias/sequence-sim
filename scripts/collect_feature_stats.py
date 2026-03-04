#!/usr/bin/env python3
"""Collect feature statistics across many games to determine scale factors.

Runs SmartAgent vs a mixed opponent pool, extracting expert features (45 dims)
every turn for both teams.  Outputs min/max/mean/std/p95/p99 per feature.

Usage:
    python scripts/collect_feature_stats.py --games 500 --workers 2
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _play_and_collect(args: tuple[int, int]) -> np.ndarray:
    """Play one game and return all feature vectors observed.

    Returns array of shape (num_observations, 45).
    """
    seed, num_teams = args
    from sequence.agents.expert.features import extract_expert_features
    from sequence.agents.smart_agent import SmartAgent
    from sequence.agents.greedy_agent import GreedyAgent
    from sequence.agents.defensive_agent import DefensiveAgent
    from sequence.agents.lookahead_agent import LookaheadAgent
    from sequence.core.board import Board
    from sequence.core.card_tracker import CardTracker
    from sequence.core.deck import Deck
    from sequence.core.game import GameConfig
    from sequence.core.game_state import GameState
    from sequence.core.types import TeamId
    from sequence.scoring.scoring_function import DEFENSIVE_WEIGHTS, ScoringFunction

    config = GameConfig(seed=seed, max_turns=300)

    # Pick opponent based on seed
    pick = seed % 4
    if pick == 0:
        opp_factory = lambda: GreedyAgent()
    elif pick == 1:
        opp_factory = lambda: DefensiveAgent()
    elif pick == 2:
        opp_factory = lambda: SmartAgent(use_lookahead=False)
    else:
        opp_factory = lambda: LookaheadAgent(
            depth=2,
            scoring_fn=ScoringFunction(DEFENSIVE_WEIGHTS),
            max_actions=10,
        )

    if seed % 2 == 0:
        factories = [lambda: SmartAgent(use_lookahead=True), opp_factory]
    else:
        factories = [opp_factory, lambda: SmartAgent(use_lookahead=True)]

    # Initialize game state (mirrors Game.play())
    board = Board()
    deck = Deck(seed=config.seed)
    hands: dict[int, list] = {}
    for t in range(num_teams):
        hand = []
        for _ in range(config.hand_size):
            card = deck.draw()
            if card is not None:
                hand.append(card)
        hands[t] = hand

    state = GameState(
        board=board,
        hands=hands,
        deck=deck,
        current_team=TeamId(0),
        num_teams=num_teams,
        sequences_to_win=config.sequences_to_win,
    )

    agents = [f() for f in factories]
    trackers = {t: CardTracker(TeamId(t), num_teams) for t in range(num_teams)}

    for i, agent in enumerate(agents):
        agent.notify_game_start(TeamId(i), config)

    all_features: list[np.ndarray] = []

    for turn in range(config.max_turns):
        winner = state.is_terminal()
        if winner is not None:
            break

        team = state.current_team
        legal = state.get_legal_actions(team)
        if not legal:
            break

        # Extract features for both teams
        for t in range(num_teams):
            tid = TeamId(t)
            feats = extract_expert_features(state, tid, tracker=trackers[t])
            all_features.append(feats)

        # Agent chooses action
        visible_state = state.get_visible_state(team)
        action = agents[team.value].choose_action(visible_state, legal)

        # Notify all agents and trackers
        for a in agents:
            a.notify_action(action, team)
        for t in range(num_teams):
            trackers[t].on_action(action, team)

        state = state.apply_action(action)

    if all_features:
        return np.array(all_features)
    return np.empty((0, 47))


def main():
    parser = argparse.ArgumentParser(description="Collect feature statistics")
    parser.add_argument("--games", type=int, default=500, help="Number of games")
    parser.add_argument("--workers", type=int, default=2, help="Parallel workers")
    parser.add_argument(
        "--output",
        type=str,
        default="data/feature_stats.json",
        help="Output file",
    )
    args = parser.parse_args()

    from concurrent.futures import ProcessPoolExecutor, as_completed
    from sequence.scoring.scoring_function import FEATURE_NAMES

    num_teams = 2
    tasks = [(seed, num_teams) for seed in range(args.games)]

    print(f"Collecting features from {args.games} games with {args.workers} workers...")

    all_data: list[np.ndarray] = []

    if args.workers <= 1:
        for i, task in enumerate(tasks):
            result = _play_and_collect(task)
            if result.shape[0] > 0:
                all_data.append(result)
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{args.games} games done")
    else:
        done = 0
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_play_and_collect, t): t for t in tasks}
            for future in as_completed(futures):
                result = future.result()
                if result.shape[0] > 0:
                    all_data.append(result)
                done += 1
                if done % 50 == 0:
                    print(f"  {done}/{args.games} games done")

    combined = np.concatenate(all_data, axis=0)
    print(f"\nTotal observations: {combined.shape[0]}")

    # Compute statistics
    stats = {}
    for i, name in enumerate(FEATURE_NAMES):
        col = combined[:, i]
        stats[name] = {
            "index": i,
            "min": float(np.min(col)),
            "max": float(np.max(col)),
            "mean": float(np.mean(col)),
            "std": float(np.std(col)),
            "p95": float(np.percentile(col, 95)),
            "p99": float(np.percentile(col, 99)),
        }

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nStats saved to: {output_path}")
    print(f"\n{'Feature':<35} {'min':>8} {'max':>8} {'mean':>8} {'p95':>8} {'p99':>8}")
    print("-" * 85)
    for name, s in stats.items():
        print(
            f"{name:<35} {s['min']:>8.2f} {s['max']:>8.2f} "
            f"{s['mean']:>8.2f} {s['p95']:>8.2f} {s['p99']:>8.2f}"
        )

    # Print suggested scales (p99 clamped >= 1.0)
    print("\n\nSuggested FEATURE_SCALES_47 (p99 clamped >= 1.0):")
    print("FEATURE_SCALES_47 = np.array([")
    for name, s in stats.items():
        scale = max(1.0, s["p99"])
        print(f"    {scale:.1f},  # {name}")
    print("], dtype=np.float64)")


if __name__ == "__main__":
    main()
