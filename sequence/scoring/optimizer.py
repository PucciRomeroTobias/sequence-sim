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
    ScoringWeights,
)

NUM_WEIGHTS = len(fields(ScoringWeights))


def _default_opponent_factory() -> object:
    """Create a default GreedyAgent opponent. Deferred import."""
    from ..agents.greedy_agent import GreedyAgent

    return GreedyAgent()


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
        game = Game(
            agent_factories=[
                lambda w=w: ScorerAgent(w),
                _default_opponent_factory,
            ],
            config=GameConfig(seed=seed, max_turns=300),
        )
        record = game.play()
        if record.winner == 0:
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
    ) -> None:
        self.population_size = population_size
        self.num_generations = num_generations
        self.games_per_eval = games_per_eval
        self.opponent_factory = opponent_factory or _default_opponent_factory
        self.num_workers = num_workers
        self.mutation_rate = mutation_rate
        self.mutation_sigma = mutation_sigma
        self.crossover_rate = crossover_rate
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
        """Create initial population seeded with known weight sets."""
        population: list[np.ndarray] = [
            BALANCED_WEIGHTS.to_array(),
            DEFENSIVE_WEIGHTS.to_array(),
            OFFENSIVE_WEIGHTS.to_array(),
        ]
        while len(population) < self.population_size:
            population.append(self._random_weights())
        return population

    def evaluate_fitness(self, weights: np.ndarray) -> float:
        """Evaluate a weight vector by playing games_per_eval games.

        Returns win rate as a float in [0, 1].
        """
        return _evaluate_weights_vs_greedy((weights, self.games_per_eval))

    def _evaluate_population(self, population: list[np.ndarray]) -> list[float]:
        """Evaluate fitness for the entire population, optionally in parallel."""
        args_list = [(w, self.games_per_eval) for w in population]

        if self.num_workers <= 1:
            return [_evaluate_weights_vs_greedy(a) for a in args_list]

        fitnesses: list[float | None] = [None] * len(population)
        with ProcessPoolExecutor(max_workers=self.num_workers) as pool:
            future_to_idx = {
                pool.submit(_evaluate_weights_vs_greedy, a): idx
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
        """Run the genetic algorithm and return the best weights and fitness."""
        population = self._initial_population()
        best_weights: np.ndarray = population[0].copy()
        best_fitness: float = 0.0

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
    """Optimize scoring weights using Powell's method from scipy.

    Falls back gracefully if scipy is not installed.
    """

    def __init__(
        self,
        games_per_eval: int = 50,
        opponent_factory: Callable[[], object] | None = None,
        maxiter: int = 100,
    ) -> None:
        self.games_per_eval = games_per_eval
        self.opponent_factory = opponent_factory or _default_opponent_factory
        self.maxiter = maxiter

    def _neg_win_rate(self, weights_array: np.ndarray) -> float:
        """Objective: negative win rate (for minimization)."""
        return -_evaluate_weights_vs_greedy((weights_array, self.games_per_eval))

    def optimize(
        self, initial_weights: ScoringWeights | None = None
    ) -> ScoringWeights:
        """Run Powell optimization and return the best weights found.

        Raises ImportError if scipy is not available.
        """
        try:
            from scipy.optimize import minimize
        except ImportError:
            raise ImportError(
                "scipy is required for CMAESOptimizer. "
                "Install it with: pip install scipy"
            )

        x0 = (initial_weights or BALANCED_WEIGHTS).to_array()
        result = minimize(
            self._neg_win_rate,
            x0,
            method="Powell",
            options={"maxiter": self.maxiter, "disp": True},
        )
        return ScoringWeights.from_array(result.x)
