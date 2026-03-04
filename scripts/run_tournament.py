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
from sequence.scoring.scoring_function import BALANCED_WEIGHTS
from sequence.simulation.tournament import Tournament


# Named functions (not lambdas) so they can be pickled for multiprocessing
def _make_random():
    return RandomAgent()

def _make_greedy():
    return GreedyAgent()

def _make_scorer():
    return ScorerAgent(BALANCED_WEIGHTS)

def _make_defensive():
    return DefensiveAgent()

def _make_offensive():
    return OffensiveAgent()

def _make_lookahead1():
    from sequence.scoring.scoring_function import ScoringFunction, DEFENSIVE_WEIGHTS
    return LookaheadAgent(depth=1, scoring_fn=ScoringFunction(DEFENSIVE_WEIGHTS))

def _make_lookahead2():
    from sequence.scoring.scoring_function import ScoringFunction, DEFENSIVE_WEIGHTS
    return LookaheadAgent(depth=2, scoring_fn=ScoringFunction(DEFENSIVE_WEIGHTS))

def _make_mcts_light():
    from sequence.agents.mcts_agent import MCTSAgent
    return MCTSAgent(iterations=200, num_determinizations=5, rollout_depth=20, max_root_actions=12)

def _make_smart():
    from sequence.agents.smart_agent import SmartAgent
    return SmartAgent(use_lookahead=True, lookahead_candidates=5)

def _make_mcts_informed():
    from sequence.agents.mcts_agent import MCTSAgent
    return MCTSAgent(
        iterations=200, num_determinizations=5, rollout_depth=20,
        max_root_actions=12, use_informed_determinization=True,
        use_heuristic_rollout=True,
    )

def _make_expert():
    from sequence.agents.expert_agent import ExpertAgent
    return ExpertAgent(use_lookahead=True, lookahead_candidates=7)

def _make_expert_optimized():
    import json
    from pathlib import Path as _P
    from sequence.agents.expert_agent import ExpertAgent
    from sequence.scoring.scoring_function import ScoringWeights
    from sequence.scoring.normalization import FEATURE_SCALES_47
    weights_path = _P("data/weights/expert_v2_stage2.json")
    if not weights_path.exists():
        weights_path = _P("data/weights/expert_v2_stage1.json")
    if not weights_path.exists():
        weights_path = _P("data/weights/expert_stage2.json")
    if not weights_path.exists():
        weights_path = _P("data/weights/expert_stage1.json")
    with open(weights_path) as f:
        w = ScoringWeights.from_dict(json.load(f))
    return ExpertAgent(
        weights=w,
        scale_factors=FEATURE_SCALES_47,
        use_lookahead=True,
        lookahead_candidates=7,
    )


def _make_neural():
    from sequence.agents.neural_agent import NeuralAgent
    return NeuralAgent("data/nn/model.pt", use_lookahead=True)


def _make_neural_norm():
    from sequence.agents.neural_agent import NeuralAgent
    return NeuralAgent("data/nn/model_norm.pt", use_lookahead=True)


def _make_lgbm():
    from sequence.agents.lgbm_agent import LGBMAgent
    return LGBMAgent("data/lgbm/model.txt")

def _make_hybrid():
    from sequence.agents.lgbm_agent import HybridAgent
    return HybridAgent("data/lgbm/model.txt")


AGENT_REGISTRY = {
    "random": _make_random,
    "greedy": _make_greedy,
    "scorer": _make_scorer,
    "defensive": _make_defensive,
    "offensive": _make_offensive,
    "lookahead1": _make_lookahead1,
    "lookahead2": _make_lookahead2,
    "mcts": _make_mcts_light,
    "smart": _make_smart,
    "mcts_informed": _make_mcts_informed,
    "expert": _make_expert,
    "expert_optimized": _make_expert_optimized,
}

# Register neural agent only if torch is available
try:
    import torch  # noqa: F401
    AGENT_REGISTRY["neural"] = _make_neural
    AGENT_REGISTRY["neural_norm"] = _make_neural_norm
except ImportError:
    pass

# Register LGBM agent only if lightgbm is available and model exists
try:
    import lightgbm  # noqa: F401
    from pathlib import Path as _Path
    if _Path("data/lgbm/model.txt").exists():
        AGENT_REGISTRY["lgbm"] = _make_lgbm
        AGENT_REGISTRY["hybrid"] = _make_hybrid
except ImportError:
    pass


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
                show_progress=False,
            )
            result = tournament.run()

            # Count wins by game index: even games = [a,b], odd games = [b,a]
            wins_a = 0
            wins_b = 0
            for game_idx, r in enumerate(result.records):
                if r.winner is not None:
                    # swap_sides=True: even-index games have a=team0, b=team1
                    # odd-index games have b=team0, a=team1
                    if game_idx % 2 == 0:
                        if r.winner == 0:
                            wins_a += 1
                        else:
                            wins_b += 1
                    else:
                        if r.winner == 0:
                            wins_b += 1
                        else:
                            wins_a += 1
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
