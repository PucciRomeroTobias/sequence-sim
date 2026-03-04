#!/usr/bin/env python3
"""Validate that normalized-space weights produce identical scores to raw weights.

The mathematical identity:
    score_old = dot(w_raw, features_raw)
    w_norm = w_raw * scales
    w_effective = w_norm / scales = w_raw  (exact roundtrip)
    score_new = dot(w_effective, features_raw) == score_old

This script verifies the identity holds to within floating-point tolerance
across real game states.

Usage:
    python scripts/validate_normalization.py --games 10
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    parser = argparse.ArgumentParser(description="Validate normalization roundtrip")
    parser.add_argument("--games", type=int, default=10, help="Number of games")
    args = parser.parse_args()

    from sequence.agents.expert.features import extract_expert_features
    from sequence.agents.smart_agent import SmartAgent
    from sequence.agents.greedy_agent import GreedyAgent
    from sequence.core.board import Board
    from sequence.core.card_tracker import CardTracker
    from sequence.core.deck import Deck
    from sequence.core.game import GameConfig
    from sequence.core.game_state import GameState
    from sequence.core.types import TeamId
    from sequence.scoring.normalization import FEATURE_SCALES_47, to_normalized_space
    from sequence.scoring.scoring_function import SMART_WEIGHTS, ScoringFunction

    # Old-style: raw weights, 35 base features only
    sf_old = ScoringFunction(SMART_WEIGHTS)

    # New-style: transform to normalized space, then ScoringFunction pre-divides back
    raw_35 = SMART_WEIGHTS.to_array()[:35]
    w_norm_35 = to_normalized_space(raw_35, FEATURE_SCALES_47[:35])

    # Build a ScoringWeights with normalized values for the 35 base features
    from sequence.scoring.scoring_function import ScoringWeights
    w_norm_full = np.zeros(47)
    w_norm_full[:35] = w_norm_35
    weights_norm = ScoringWeights.from_array(w_norm_full)

    # ScoringFunction with scale_factors should pre-divide back to raw
    sf_new = ScoringFunction(
        weights_norm,
        use_expert_features=False,
        scale_factors=FEATURE_SCALES_47,
    )

    max_diff = 0.0
    total_states = 0
    num_teams = 2

    for seed in range(args.games):
        config = GameConfig(seed=seed, max_turns=300)
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

        agents = [SmartAgent(use_lookahead=False), GreedyAgent()]
        trackers = {t: CardTracker(TeamId(t), num_teams) for t in range(num_teams)}
        for i, a in enumerate(agents):
            a.notify_game_start(TeamId(i), config)

        for turn in range(config.max_turns):
            winner = state.is_terminal()
            if winner is not None:
                break

            team = state.current_team
            legal = state.get_legal_actions(team)
            if not legal:
                break

            # Compare scores
            for t in range(num_teams):
                tid = TeamId(t)
                score_old = sf_old.evaluate(state, tid, tracker=trackers[t])
                score_new = sf_new.evaluate(state, tid, tracker=trackers[t])
                diff = abs(score_old - score_new)
                max_diff = max(max_diff, diff)
                total_states += 1

                if diff > 1e-6:
                    print(
                        f"MISMATCH at game {seed}, turn {turn}, team {t}: "
                        f"old={score_old:.8f} new={score_new:.8f} diff={diff:.2e}"
                    )

            # Play
            visible = state.get_visible_state(team)
            action = agents[team.value].choose_action(visible, legal)
            for a in agents:
                a.notify_action(action, team)
            for t in range(num_teams):
                trackers[t].on_action(action, team)
            state = state.apply_action(action)

    print(f"\nValidated {total_states} state evaluations across {args.games} games")
    print(f"Max absolute difference: {max_diff:.2e}")

    if max_diff < 1e-6:
        print("PASS: Normalization roundtrip is exact within tolerance")
    else:
        print("FAIL: Differences exceed tolerance!")
        sys.exit(1)


if __name__ == "__main__":
    main()
