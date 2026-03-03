"""Scoring weight optimizers using genetic algorithm and CMA-ES."""

from __future__ import annotations

import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import fields
from typing import Callable

import numpy as np

from .scoring_function import (
    BALANCED_WEIGHTS,
    DEFENSIVE_WEIGHTS,
    OFFENSIVE_WEIGHTS,
    SMART_WEIGHTS,
    ScoringWeights,
)

NUM_WEIGHTS = len(fields(ScoringWeights))

# Custom opponents for self-play iterations (module-level for pickling)
_custom_opponents: list[np.ndarray] = []


def add_custom_opponent(weights: ScoringWeights) -> None:
    """Add a SmartAgent with given weights to the mixed opponent pool."""
    _custom_opponents.append(weights.to_array())


def clear_custom_opponents() -> None:
    """Clear the custom opponent pool."""
    _custom_opponents.clear()


def _default_opponent_factory() -> object:
    """Create a default GreedyAgent opponent. Deferred import."""
    from ..agents.greedy_agent import GreedyAgent

    return GreedyAgent()


def _make_mixed_opponent(seed: int) -> object:
    """Create a mixed opponent based on seed for diverse evaluation.

    If custom opponents exist, they get priority slots.
    Remaining distribution: 40% LookaheadAgent(depth=2), 20% SmartAgent, 20% GreedyAgent, 20% DefensiveAgent.
    """
    # Give custom opponents priority: cycle through them first
    num_custom = len(_custom_opponents)
    if num_custom > 0:
        total_slots = 5 + num_custom
        pick = seed % total_slots
        if pick < num_custom:
            from ..agents.smart_agent import SmartAgent
            w = ScoringWeights.from_array(_custom_opponents[pick])
            return SmartAgent(weights=w, use_lookahead=False)
        pick = (pick - num_custom) % 5
    else:
        pick = seed % 5

    if pick <= 1:
        from ..agents.lookahead_agent import LookaheadAgent
        from .scoring_function import ScoringFunction
        return LookaheadAgent(
            depth=2,
            scoring_fn=ScoringFunction(DEFENSIVE_WEIGHTS),
            max_actions=10,
        )
    elif pick == 2:
        from ..agents.smart_agent import SmartAgent
        return SmartAgent(use_lookahead=False)
    elif pick == 3:
        from ..agents.greedy_agent import GreedyAgent
        return GreedyAgent()
    else:
        from ..agents.defensive_agent import DefensiveAgent
        return DefensiveAgent()


def _evaluate_weights_vs_greedy(args: tuple[np.ndarray, int]) -> float:
    """Evaluate a weight vector by playing games against GreedyAgent.

    Module-level function so it can be pickled for multiprocessing.
    """
    from ..agents.scorer_agent import ScorerAgent
    from ..core.game import Game, GameConfig

    weights_array, games_per_eval = args
    w = ScoringWeights.from_array(weights_array)
    wins = 0
    for seed in range(games_per_eval):
        # Alternate sides to avoid first-player bias
        if seed % 2 == 0:
            factories = [
                lambda w=w: ScorerAgent(w),
                _default_opponent_factory,
            ]
            win_idx = 0
        else:
            factories = [
                _default_opponent_factory,
                lambda w=w: ScorerAgent(w),
            ]
            win_idx = 1
        game = Game(
            agent_factories=factories,
            config=GameConfig(seed=seed, max_turns=300),
        )
        record = game.play()
        if record.winner == win_idx:
            wins += 1
    return wins / games_per_eval


def _evaluate_weights_smart(args: tuple[np.ndarray, int, bool] | tuple[np.ndarray, int, bool, bool]) -> float:
    """Evaluate a weight vector using SmartAgent against mixed opponents.

    Module-level function so it can be pickled for multiprocessing.
    Args tuple: (weights_array, games_per_eval, use_mixed[, use_lookahead])
    """
    from ..agents.smart_agent import SmartAgent
    from ..core.game import Game, GameConfig

    if len(args) == 4:
        weights_array, games_per_eval, use_mixed, use_lookahead = args
    else:
        weights_array, games_per_eval, use_mixed = args  # type: ignore[misc]
        use_lookahead = False
    w = ScoringWeights.from_array(weights_array)
    wins = 0
    for seed in range(games_per_eval):
        if use_mixed:
            opp_factory = lambda s=seed: _make_mixed_opponent(s)
        else:
            opp_factory = _default_opponent_factory
        # Alternate sides to avoid first-player bias
        if seed % 2 == 0:
            factories = [
                lambda w=w, la=use_lookahead: SmartAgent(weights=w, use_lookahead=la),
                opp_factory,
            ]
            win_idx = 0
        else:
            factories = [
                opp_factory,
                lambda w=w, la=use_lookahead: SmartAgent(weights=w, use_lookahead=la),
            ]
            win_idx = 1
        game = Game(
            agent_factories=factories,
            config=GameConfig(seed=seed, max_turns=300),
        )
        record = game.play()
        if record.winner == win_idx:
            wins += 1
    return wins / games_per_eval


class GeneticOptimizer:
    """Optimize scoring weights using a genetic algorithm."""

    def __init__(
        self,
        population_size: int = 30,
        num_generations: int = 50,
        games_per_eval: int = 50,
        opponent_factory: Callable[[], object] | None = None,
        num_workers: int = 4,
        mutation_rate: float = 0.2,
        mutation_sigma: float = 0.3,
        crossover_rate: float = 0.7,
        seed: int | None = None,
        use_smart_agent: bool = False,
        use_mixed_opponents: bool = False,
        patience: int = 5,
        early_stop_epsilon: float = 0.001,
        use_lookahead: bool = False,
    ) -> None:
        self.population_size = population_size
        self.num_generations = num_generations
        self.games_per_eval = games_per_eval
        self.opponent_factory = opponent_factory or _default_opponent_factory
        self.num_workers = num_workers
        self.mutation_rate = mutation_rate
        self.mutation_sigma = mutation_sigma
        self.crossover_rate = crossover_rate
        self.use_smart_agent = use_smart_agent
        self.use_mixed_opponents = use_mixed_opponents
        self.patience = patience
        self.early_stop_epsilon = early_stop_epsilon
        self.use_lookahead = use_lookahead
        self._rng = random.Random(seed)
        self._np_rng = np.random.RandomState(seed)

    def _random_weights(self) -> np.ndarray:
        """Generate a random weight vector."""
        arr = self._np_rng.uniform(-10, 10, size=NUM_WEIGHTS)
        # completed_sequences should be positive and large
        arr[0] = abs(arr[0]) * 10
        # opp_completed_sequences should be negative
        arr[4] = -abs(arr[4]) * 10
        return arr

    def _initial_population(self) -> list[np.ndarray]:
        """Create initial population seeded with known weight sets.

        After the 4 presets, fills with Gaussian perturbations of SMART_WEIGHTS
        (the best known weights) plus 2 random individuals for diversity.
        """
        population: list[np.ndarray] = [
            BALANCED_WEIGHTS.to_array(),
            DEFENSIVE_WEIGHTS.to_array(),
            OFFENSIVE_WEIGHTS.to_array(),
            SMART_WEIGHTS.to_array(),
        ]
        smart_arr = SMART_WEIGHTS.to_array()
        # Fill most remaining slots with perturbations of SMART_WEIGHTS
        while len(population) < self.population_size - 2:
            sigma = 0.1 * np.maximum(np.abs(smart_arr), 1.0)
            perturbed = smart_arr + self._np_rng.normal(0, 1, size=NUM_WEIGHTS) * sigma
            population.append(perturbed)
        # Keep 2 random individuals for diversity
        while len(population) < self.population_size:
            population.append(self._random_weights())
        return population

    def evaluate_fitness(self, weights: np.ndarray) -> float:
        """Evaluate a weight vector by playing games_per_eval games.

        Returns win rate as a float in [0, 1].
        """
        if self.use_smart_agent:
            return _evaluate_weights_smart(
                (weights, self.games_per_eval, self.use_mixed_opponents, self.use_lookahead)
            )
        return _evaluate_weights_vs_greedy((weights, self.games_per_eval))

    def _evaluate_population(self, population: list[np.ndarray]) -> list[float]:
        """Evaluate fitness for the entire population, optionally in parallel."""
        if self.use_smart_agent:
            args_list = [
                (w, self.games_per_eval, self.use_mixed_opponents, self.use_lookahead)
                for w in population
            ]
            eval_fn = _evaluate_weights_smart
        else:
            args_list = [(w, self.games_per_eval) for w in population]
            eval_fn = _evaluate_weights_vs_greedy

        if self.num_workers <= 1:
            return [eval_fn(a) for a in args_list]

        fitnesses: list[float | None] = [None] * len(population)
        with ProcessPoolExecutor(max_workers=self.num_workers) as pool:
            future_to_idx = {
                pool.submit(eval_fn, a): idx
                for idx, a in enumerate(args_list)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                fitnesses[idx] = future.result()
        return [f for f in fitnesses]  # type: ignore[misc]

    def select(
        self, population: list[np.ndarray], fitnesses: list[float]
    ) -> list[np.ndarray]:
        """Tournament selection with k=3."""
        parents: list[np.ndarray] = []
        k = 3
        for _ in range(len(population)):
            candidates = self._rng.sample(range(len(population)), k=min(k, len(population)))
            best = max(candidates, key=lambda i: fitnesses[i])
            parents.append(population[best].copy())
        return parents

    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """BLX-alpha blend crossover."""
        alpha = 0.5
        child = np.empty(NUM_WEIGHTS)
        for i in range(NUM_WEIGHTS):
            lo = min(parent1[i], parent2[i])
            hi = max(parent1[i], parent2[i])
            span = hi - lo
            child[i] = self._np_rng.uniform(lo - alpha * span, hi + alpha * span)
        return child

    def mutate(self, weights: np.ndarray) -> np.ndarray:
        """Gaussian mutation applied per weight with given probability."""
        mutated = weights.copy()
        for i in range(NUM_WEIGHTS):
            if self._np_rng.random() < self.mutation_rate:
                mutated[i] += self._np_rng.normal(0, self.mutation_sigma) * max(1.0, abs(mutated[i]))
        return mutated

    def optimize(self) -> tuple[ScoringWeights, float]:
        """Run the genetic algorithm and return the best weights and fitness.

        Supports early stopping: if best_fitness doesn't improve by more than
        early_stop_epsilon for `patience` consecutive generations, stops early.
        """
        population = self._initial_population()
        best_weights: np.ndarray = population[0].copy()
        best_fitness: float = 0.0
        prev_best: float = float("-inf")
        stagnation_count: int = 0

        for gen in range(self.num_generations):
            fitnesses = self._evaluate_population(population)

            # Track best
            gen_best_idx = int(np.argmax(fitnesses))
            gen_best_fitness = fitnesses[gen_best_idx]
            if gen_best_fitness > best_fitness:
                best_fitness = gen_best_fitness
                best_weights = population[gen_best_idx].copy()

            print(
                f"Generation {gen + 1}/{self.num_generations}: "
                f"best={gen_best_fitness:.3f}  avg={np.mean(fitnesses):.3f}  "
                f"overall_best={best_fitness:.3f}"
            )

            # Early stopping check
            if best_fitness - prev_best > self.early_stop_epsilon:
                stagnation_count = 0
            else:
                stagnation_count += 1
            prev_best = best_fitness

            if stagnation_count >= self.patience:
                print(
                    f"Early stopping: no improvement for {self.patience} generations"
                )
                break

            # Selection
            parents = self.select(population, fitnesses)

            # Create next generation
            next_population: list[np.ndarray] = []
            # Elitism: keep the best individual
            next_population.append(best_weights.copy())

            while len(next_population) < self.population_size:
                p1, p2 = self._rng.sample(parents, 2)
                if self._np_rng.random() < self.crossover_rate:
                    child = self.crossover(p1, p2)
                else:
                    child = p1.copy()
                child = self.mutate(child)
                next_population.append(child)

            population = next_population

        return ScoringWeights.from_array(best_weights), best_fitness


class CMAESOptimizer:
    """Optimize scoring weights using CMA-ES (Covariance Matrix Adaptation).

    Requires the ``cma`` package (``pip install cma``).
    """

    def __init__(
        self,
        games_per_eval: int = 50,
        num_workers: int = 4,
        maxiter: int = 100,
        sigma0: float = 5.0,
        use_smart_agent: bool = False,
        use_mixed_opponents: bool = False,
        use_lookahead: bool = False,
    ) -> None:
        self.games_per_eval = games_per_eval
        self.num_workers = num_workers
        self.maxiter = maxiter
        self.sigma0 = sigma0
        self.use_smart_agent = use_smart_agent
        self.use_mixed_opponents = use_mixed_opponents
        self.use_lookahead = use_lookahead

    def optimize(
        self, initial_weights: ScoringWeights | None = None
    ) -> tuple[ScoringWeights, float]:
        """Run CMA-ES optimization and return the best weights and fitness.

        Raises ImportError if the cma package is not available.
        """
        try:
            import cma
        except ImportError:
            raise ImportError(
                "cma is required for CMAESOptimizer. "
                "Install it with: pip install cma"
            )

        from math import log

        x0 = (initial_weights or SMART_WEIGHTS).to_array()
        popsize = max(8, 4 + int(3 * log(NUM_WEIGHTS)))

        es = cma.CMAEvolutionStrategy(
            x0,
            self.sigma0,
            {
                "maxiter": self.maxiter,
                "popsize": popsize,
                "verbose": 1,
            },
        )

        if self.use_smart_agent:
            eval_fn = _evaluate_weights_smart
        else:
            eval_fn = _evaluate_weights_vs_greedy

        best_fitness = 0.0
        best_weights = x0.copy()

        while not es.stop():
            solutions = es.ask()

            if self.use_smart_agent:
                args_list = [
                    (s, self.games_per_eval, self.use_mixed_opponents, self.use_lookahead)
                    for s in solutions
                ]
            else:
                args_list = [(s, self.games_per_eval) for s in solutions]

            # Evaluate in parallel
            if self.num_workers <= 1:
                fitnesses = [eval_fn(a) for a in args_list]
            else:
                fitnesses_arr: list[float | None] = [None] * len(solutions)
                with ProcessPoolExecutor(max_workers=self.num_workers) as pool:
                    future_to_idx = {
                        pool.submit(eval_fn, a): idx
                        for idx, a in enumerate(args_list)
                    }
                    for future in as_completed(future_to_idx):
                        idx = future_to_idx[future]
                        fitnesses_arr[idx] = future.result()
                fitnesses = [f for f in fitnesses_arr]  # type: ignore[misc]

            # CMA-ES minimizes, so negate win rates
            es.tell(solutions, [-f for f in fitnesses])
            es.disp()

            gen_best = max(fitnesses)
            if gen_best > best_fitness:
                best_fitness = gen_best
                best_idx = fitnesses.index(gen_best)
                best_weights = solutions[best_idx].copy()

        return ScoringWeights.from_array(best_weights), best_fitness
