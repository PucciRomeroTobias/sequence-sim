# CLAUDE.md — sequence-sim

Simulador del juego de mesa Sequence con agentes IA (MCTS, scoring genético, card counting).

## Estructura del proyecto

```
sequence/
  core/         # Motor del juego: board, card, deck, game_state, card_tracker, types, actions
  agents/       # Agentes: random, greedy, defensive, lookahead, smart, mcts
  scoring/      # Features (33), scoring function, weights, optimizer genético
  analysis/     # Explainer, estadísticas, heatmaps, game phases
  simulation/   # Runner, tournament, dataset generation
scripts/        # CLIs: run_tournament, optimize_weights, explain_strategy, launch_gui
tests/          # Tests con pytest
```

## Cómo correr

```bash
# Tests
python -m pytest tests/ -x -q

# Torneo entre agentes
python scripts/run_tournament.py --agents random,greedy,defensive,smart --games 100

# Optimizar pesos del scoring
python scripts/optimize_weights.py --smart --mixed --generations 20 --population 20

# Reporte de estrategia
python scripts/explain_strategy.py --preset smart
```

## Dependencias

- Python >=3.11, numpy
- Dev: pytest, pytest-timeout
- Sim: tqdm
- Analysis: pandas, scipy
- Instalar: `pip install -e ".[all]"`

## Convenciones del código

- 33 features en `scoring/features.py` — `extract_features(state, team, tracker=None)`
- `CardTracker` es opcional: si se pasa, habilita las 11 features de card counting; si no, defaultean a 0
- `ScoringWeights` es backward-compatible: `from_dict()` filtra keys desconocidos, `from_array()` acepta arrays cortos
- Agentes implementan `Agent` (base class en `agents/base.py`): `choose_action()`, `notify_game_start()`, `notify_action()`
- El board tiene 10x10, layout fijo en `core/board.py`, 192 líneas posibles en `ALL_LINES`
- `POSITION_TO_LINES[pos]` da los índices de líneas que pasan por cada posición
- Cada carta no-jack aparece 2 veces en el tablero y 2 veces en el mazo (104 cartas total)

## Estado actual

- Fases A-D completadas: CardTracker, 33 features, SmartAgent, MCTS mejorado
- Fase E completada: optimizer, tournament runner, scripts
- Fase F completada: estrategias avanzadas (SE-based scoring, anchor/overlap, game phase, clustering, jack timing)
- Pendiente: optimizar pesos con las 33 features
- 138 tests pasan

## Nota sobre tests

Este proyecto tiene unit tests existentes (anteriores al CLAUDE.md padre). Mantenerlos como safety net pero para trabajo nuevo, verificar ejecutando scripts/torneos directamente.
