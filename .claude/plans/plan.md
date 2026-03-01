# Sequence Board Game Simulator - Plan

## Context

Queremos aprender a jugar mejor a Sequence construyendo un simulador que:
1. Juegue miles de partidas entre diferentes estrategias
2. Use MCTS como "jugador experto artificial" para generar un dataset de jugadas
3. Derive una **funcion de scoring interpretable** (pesos por factor) que traduzca a consejos humanos
4. Tenga una GUI simple para visualizar partidas y heatmaps de "valor" por casilla

No existe base de datos de partidas profesionales de Sequence, por lo que generamos nuestro propio dataset con MCTS vs MCTS.

**Ubicacion del proyecto**: `/Users/tpucci/sequence-sim/`

---

## Estructura del Proyecto

```
sequence-sim/
├── pyproject.toml
├── sequence/
│   ├── __init__.py
│   ├── core/                    # Motor del juego (0 deps externas)
│   │   ├── __init__.py
│   │   ├── types.py             # Suit, Rank, TeamId, Position, CORNERS
│   │   ├── card.py              # Card (frozen dataclass, hashable)
│   │   ├── board.py             # Board: chips 10x10 numpy int8, sequence detection
│   │   ├── deck.py              # Deck: 104 cartas, draw/discard/reshuffle
│   │   ├── actions.py           # Action: card + position + type (place/remove/dead)
│   │   ├── game_state.py        # GameState: legal actions, apply_action, is_terminal
│   │   └── game.py              # Game: orquestador de turnos, genera GameRecord
│   │
│   ├── agents/                  # Estrategias (todas implementan Agent ABC)
│   │   ├── __init__.py
│   │   ├── base.py              # Agent ABC: choose_action(state, legal_actions) -> Action
│   │   ├── random_agent.py      # Baseline: elige al azar
│   │   ├── greedy_agent.py      # Mejor movimiento inmediato (sin lookahead)
│   │   ├── scorer_agent.py      # Evalua scoring_function por cada accion
│   │   ├── lookahead_agent.py   # Minimax alpha-beta depth 1 y 2
│   │   ├── mcts_agent.py        # MCTS con determinizacion (info imperfecta)
│   │   ├── defensive_agent.py   # ScorerAgent con pesos defensivos
│   │   └── offensive_agent.py   # ScorerAgent con pesos ofensivos
│   │
│   ├── scoring/                 # Funcion de evaluacion + optimizador
│   │   ├── __init__.py
│   │   ├── features.py          # Extrae ~17 features del GameState
│   │   ├── scoring_function.py  # score = dot(weights, features)
│   │   └── optimizer.py         # Algoritmo genetico + CMA-ES para tunar pesos
│   │
│   ├── simulation/              # Torneos y generacion de datos
│   │   ├── __init__.py
│   │   ├── runner.py            # Corre 1 partida (pickle-friendly)
│   │   ├── tournament.py        # N partidas en paralelo (ProcessPoolExecutor)
│   │   └── dataset.py           # Escribe/lee GameRecords en JSONL
│   │
│   ├── analysis/                # Analisis post-hoc
│   │   ├── __init__.py
│   │   ├── statistics.py        # Win rates, Elo, intervalos de confianza
│   │   ├── heatmaps.py          # Mapas de calor 10x10 (frecuencia, contribucion)
│   │   └── explainer.py         # Traduce pesos optimos -> consejos legibles
│   │
│   └── gui/                     # Visualizacion (tkinter)
│       ├── __init__.py
│       ├── app.py               # Ventana principal: live / replay / analysis
│       ├── board_canvas.py      # Renderiza tablero, fichas, highlights
│       ├── heatmap_view.py      # Overlay de colores por valor
│       └── replay_view.py       # Controles: play/pause/step/slider
│
├── scripts/
│   ├── generate_dataset.py      # CLI: MCTS vs MCTS -> JSONL
│   ├── optimize_weights.py      # CLI: tunar pesos con GA
│   ├── run_tournament.py        # CLI: comparar estrategias
│   ├── explain_strategy.py      # CLI: generar consejos humanos
│   └── launch_gui.py            # CLI: abrir GUI
│
├── tests/
│   ├── test_card.py
│   ├── test_board.py
│   ├── test_deck.py
│   ├── test_game.py
│   ├── test_agents.py
│   ├── test_scoring.py
│   └── test_mcts.py
│
└── data/                        # Generado (gitignored)
    ├── datasets/
    ├── weights/
    └── results/
```

**Grafo de dependencias**:
```
core/ <--- agents/ <--- simulation/ <--- scripts/
  ^          ^              ^
  |          |              |
  +--- scoring/        analysis/
                            ^
                            |
                          gui/
```

---

## FASE 1: Motor del Juego (core/)

**Objetivo**: Tablero funcional con reglas completas de Sequence.

### Archivos a crear (en orden):

**1.1 `core/types.py`** — Tipos base
- `Suit` enum (S, H, D, C)
- `Rank` enum (2-10, J, Q, K, A)
- `TeamId` IntEnum (0, 1, 2)
- `Position` NamedTuple (row, col)
- `CORNERS` frozenset de las 4 esquinas

**1.2 `core/card.py`** — Cartas
- `Card` frozen dataclass con `rank`, `suit`
- Properties: `is_one_eyed_jack` (JH, JS), `is_two_eyed_jack` (JD, JC)
- `Card.from_str("10D")` parser
- Hashable para usar como key de dict

**1.3 `core/board.py`** — Tablero
- Layout oficial 10x10 hardcodeado (del repo de GitHub)
- `CARD_TO_POSITIONS: dict[Card, list[Position]]` precalculado
- `Board` clase: `_chips` numpy int8 10x10 (-1=vacio, 0/1/2=equipo, 3=esquina)
- `place_chip()`, `remove_chip()`, `is_empty()`, `is_part_of_sequence()`
- `check_new_sequences(pos, team)`: chequea 4 direcciones desde pos, busca 5 en linea
- `count_sequences(team)`, `copy()` (numpy copy, rapido)

**1.4 `core/deck.py`** — Mazo
- 104 cartas (2 mazos completos), seeded RNG
- `draw()`, `discard()`, reshuffle automatico cuando se acaba
- `copy(rng)` para clonar en MCTS

**1.5 `core/actions.py`** — Acciones
- `ActionType` enum: PLACE, REMOVE, DEAD_CARD_DISCARD
- `Action` frozen dataclass: card, position, action_type

**1.6 `core/game_state.py`** — Estado del juego
- `GameState`: board, current_team, hands, turn_number
- `get_legal_actions(team)`: genera todas las acciones validas
  - Cartas normales -> 0-2 posiciones en tablero
  - Two-eyed Jack -> cualquier casilla vacia (usar set cacheado!)
  - One-eyed Jack -> cualquier ficha oponente no en secuencia completada
  - Dead cards -> discard
- `apply_action(action)` -> nuevo GameState
- `is_terminal()` -> TeamId ganador o None
- `get_visible_state(team)` -> copia sin manos oponentes

**1.7 `core/game.py`** — Orquestador
- `GameConfig`: num_teams, hand_size, seed
- `MoveRecord`: turn, team, action, legal_actions, hand_before, board_snapshot, card_drawn, sequences_before/after, thinking_time_ms
- `GameRecord`: game_id, seed, agents, winner, total_turns, moves[]
- `Game.play()` -> GameRecord: loop de turnos, registra todo

### Tests Fase 1:
- Sequence detection en las 4 direcciones + esquinas
- Legal action generation para cada tipo de carta
- Partida completa termina con ganador
- Misma seed = misma partida (determinismo)

---

## FASE 2: Agentes Basicos + Simulador

**Objetivo**: Poder correr partidas entre agentes simples.

### 2.1 `agents/base.py` — Interfaz
```python
class Agent(ABC):
    def choose_action(self, state: GameStateView, legal_actions: list[Action]) -> Action
    def notify_game_start(self, team, config)  # opcional
    def notify_action(self, action, team)       # opcional
```

### 2.2 `agents/random_agent.py`
- `rng.choice(legal_actions)` — baseline

### 2.3 `agents/greedy_agent.py`
- Evalua cada accion: +1000 si completa secuencia, +100 si extiende a 4, +10 si a 3, +bloqueo
- Sin lookahead, solo evalua el estado despues de cada accion

### 2.4 `simulation/runner.py`
- `run_single_game(agent_factories, config) -> GameRecord`
- Pickle-friendly (factories, no instancias)

### 2.5 `simulation/tournament.py`
- `Tournament.run()` con `ProcessPoolExecutor`
- N partidas en paralelo (usa 10 cores disponibles)
- `swap_sides=True`: cada seed se juega 2 veces intercambiando posiciones
- Progress bar con tqdm

### 2.6 `simulation/dataset.py`
- `DatasetWriter`: escribe GameRecords como JSONL (1 linea por partida)
- `DatasetReader`: lee, itera, convierte a pandas DataFrame

### Tests Fase 2:
- RandomAgent siempre devuelve accion valida
- GreedyAgent gana >65% vs RandomAgent en 200 partidas
- Tournament corre 100 partidas sin errores
- Dataset se escribe y se lee correctamente

---

## FASE 3: Sistema de Scoring

**Objetivo**: Funcion de evaluacion parametrizable con pesos tunables.

### 3.1 `scoring/features.py` — Extraccion de features

17 features por estado:

| Feature | Descripcion |
|---------|-------------|
| `completed_sequences` | Secuencias propias completadas |
| `four_in_a_row` | Lineas propias de 4 (a 1 de ganar) |
| `three_in_a_row` | Lineas propias de 3 |
| `two_in_a_row` | Lineas propias de 2 |
| `opp_completed_sequences` | Secuencias del oponente |
| `opp_four_in_a_row` | Lineas de 4 del oponente (URGENTE bloquear) |
| `opp_three_in_a_row` | Lineas de 3 del oponente |
| `chips_on_board` | Fichas propias en tablero |
| `opp_chips_on_board` | Fichas oponente en tablero |
| `center_control` | Fichas en zona central 6x6 |
| `corner_adjacency` | Fichas adyacentes a esquinas libres |
| `hand_pairs` | Cartas en mano que comparten posicion |
| `two_eyed_jacks_in_hand` | Jotas wild en mano |
| `one_eyed_jacks_in_hand` | Jotas de remocion en mano |
| `dead_cards_in_hand` | Cartas sin posicion valida |
| `shared_line_potential` | Fichas que participan en multiples secuencias posibles |
| `blocked_lines` | Lineas propias bloqueadas por oponente |

### 3.2 `scoring/scoring_function.py`
- `ScoringWeights` dataclass con los 17 pesos (serializables a JSON y numpy vector)
- `ScoringFunction.evaluate(state, team) -> float` = dot(weights, features)
- `ScoringFunction.rank_actions(state, legal_actions, team)` -> lista ordenada por score
- Perfiles pre-construidos: `DEFENSIVE_WEIGHTS`, `OFFENSIVE_WEIGHTS`, `BALANCED_WEIGHTS`

### 3.3 `agents/scorer_agent.py`
- Recibe un `ScoringFunction`, evalua cada accion legal, elige la de mayor score

### 3.4 `agents/defensive_agent.py` y `agents/offensive_agent.py`
- ScorerAgent con pesos especificos pre-configurados

### Tests Fase 3:
- Features correctas para tableros conocidos
- ScorerAgent(balanced) gana >70% vs RandomAgent
- DefensiveAgent vs OffensiveAgent: verificar que ambos ganan >50% vs random

---

## FASE 4: Agentes con Lookahead (+1, +2 turnos)

**Objetivo**: Minimax con poda alpha-beta que mira 1-2 turnos adelante.

### 4.1 `agents/lookahead_agent.py`
- `LookaheadAgent(depth=1, scoring_fn)`: minimax depth 1 (mi turno)
- `LookaheadAgent(depth=2, scoring_fn)`: minimax depth 2 (mi turno + respuesta oponente)
- Alpha-beta pruning para reducir el branching factor
- Para Two-eyed Jacks (hasta ~96 acciones): pre-filtrar top K=10 por heuristica rapida antes del minimax
- Usa `ScoringFunction.evaluate()` como funcion de evaluacion en hojas
- Manejo de info imperfecta: asume mano oponente desconocida, usa todos los movimientos legales visibles

### Tests Fase 4:
- Lookahead(1) gana >60% vs GreedyAgent
- Lookahead(2) gana >55% vs Lookahead(1)
- Tiempo por movimiento < 1s para depth=2

---

## FASE 5: Agente MCTS

**Objetivo**: Jugador "experto" artificial basado en Monte Carlo Tree Search.

### 5.1 `agents/mcts_agent.py`

**Desafio principal**: Sequence es juego de informacion imperfecta (no ves las cartas del oponente).

**Solucion**: Information Set MCTS con **determinizacion**:
1. Por cada iteracion, samplear una asignacion plausible de cartas al oponente (del pool de cartas no vistas)
2. Correr MCTS estandar sobre ese estado determinizado
3. Agregar estadisticas de visitas entre determinizaciones
4. Elegir la accion con mas visitas totales

**Parametros clave**:
- `iterations: int = 1000` (o `time_limit_ms`)
- `num_determinizations: int = 10`
- `exploration_constant: float = 1.41` (UCB1)
- `rollout_policy`: random por defecto

**MCTSNode**: state, parent, children dict, untried_actions, visits, total_value

**Flujo**: select(UCB1) -> expand -> simulate(rollout) -> backpropagate

**Optimizaciones de performance**:
- Board copy es numpy array copy (800 bytes, rapido)
- Sequence detection solo chequea desde la posicion recien jugada
- Cache de posiciones vacias (set incremental en Board)
- Rollout truncado a 50 movimientos con eval heuristica
- Progressive widening para Two-eyed Jacks (limitar branching)
- Opcionalmente: registrar `mcts_root_visits` y `mcts_root_values` por accion en el MoveRecord

### Tests Fase 5:
- MCTS(iterations=1) ~ random
- MCTS(iterations=500) gana >80% vs RandomAgent
- MCTS(iterations=1000) gana >60% vs GreedyAgent
- Determinizacion produce estados validos

---

## FASE 6: Generacion de Dataset

**Objetivo**: Crear base de datos de partidas MCTS vs MCTS.

### 6.1 `scripts/generate_dataset.py`
- Corre 1000+ partidas MCTS vs MCTS en paralelo
- Cada MoveRecord incluye:
  - Estado del tablero (10x10 int8)
  - Mano del jugador
  - Todas las acciones legales
  - Accion elegida por MCTS
  - Visitas MCTS por accion (el "confidence" del experto)
  - Secuencias antes/despues
- Guarda en `data/datasets/mcts_vs_mcts.jsonl`
- Estimacion: ~1000 partidas, ~30 movimientos/partida = ~30,000 registros de movimiento

### 6.2 Schema del dataset

Cada linea JSONL = 1 GameRecord completo con:
- `game_id`, `seed`, `winner`, `total_turns`
- `moves[]`: cada uno con `action`, `legal_actions`, `board_state`, `hand_before`, `mcts_visits`, `mcts_values`

---

## FASE 7: Optimizador de Pesos

**Objetivo**: Encontrar los pesos optimos de la funcion de scoring.

### 7.1 `scoring/optimizer.py`

**Algoritmo genetico** (principal):
1. Poblacion de 30 vectores de pesos aleatorios
2. Fitness = win rate de ScorerAgent(pesos) vs MCTS(iterations=200) en 50 partidas
3. Seleccion por torneo, crossover (blend), mutacion gaussiana
4. 50 generaciones
5. Mejor individuo = pesos optimos

**CMA-ES** (alternativa via scipy):
- `scipy.optimize.minimize(method="Powell")` sobre `-win_rate(weights)`
- Mejor para ajuste fino despues del GA

**Grid search** (para subconjuntos):
- Util para tunar 2-3 pesos especificos manteniendo el resto fijo

### 7.2 `scripts/optimize_weights.py`
- CLI que corre el optimizador y guarda los mejores pesos en `data/weights/optimized.json`

### Tests Fase 7:
- Fitness mejora a lo largo de las generaciones
- Pesos optimizados ganan >55% vs pesos default

---

## FASE 8: GUI (tkinter)

**Objetivo**: Visualizar partidas y entender que "piensa" cada estrategia.

### 8.1 `gui/board_canvas.py`
- Canvas tkinter 600x600 (60px por celda)
- Muestra: nombre de carta, fichas de color (rojo/azul/verde), esquinas doradas
- Highlights: ultima jugada, acciones legales, hover

### 8.2 `gui/heatmap_view.py`
- Overlay de colores sobre el tablero (rojo=malo, verde=bueno)
- Muestra el score de cada posicion segun la funcion de scoring activa

### 8.3 `gui/replay_view.py`
- Slider de turnos, play/pause/step forward/backward
- Velocidad configurable
- Info panel: mano, scores, turno, secuencias

### 8.4 `gui/app.py` — 3 modos:
1. **Live**: ver agentes jugar en tiempo real
2. **Replay**: cargar un GameRecord y navegar turno a turno
3. **Analysis**: heatmaps, breakdown de scoring por posicion

---

## FASE 9: Analisis y Explicacion

**Objetivo**: Traducir datos en consejos utiles para humanos.

### 9.1 `analysis/statistics.py`
- Win rates con intervalos de confianza (Wilson score)
- Elo ratings entre agentes
- Ventaja de primer jugador
- Duracion promedio de partidas

### 9.2 `analysis/heatmaps.py`
- Frecuencia de colocacion por casilla
- Correlacion casilla-victoria (win contribution)
- Atencion MCTS promedio por casilla
- Frecuencia de participacion en secuencias completadas

### 9.3 `analysis/explainer.py`
- Toma los pesos optimos y genera texto:
  - "Bloquear lineas de 4 del oponente es CRITICO (peso: -150, 3x mas importante que extender tus propias lineas de 3)"
  - "Las casillas E4-F6 tienen 2.3x mayor contribucion a la victoria"
  - "Guarda las Jotas de 2 ojos para completar secuencias"
- Rankea consejos por prioridad basada en magnitud de peso y significancia estadistica

---

## Orden de Implementacion con Agentes Paralelos

La idea es usar multiples agentes de Claude trabajando en paralelo. Aqui esta como se dividen:

### Sprint 1: Fundamentos (Fases 1-2)
- **Agente A**: `core/types.py` + `core/card.py` + `core/board.py` + `core/deck.py` + `core/actions.py`
- **Agente B**: `core/game_state.py` + `core/game.py` (depende de A, espera o usa stubs)
- **Agente C**: `agents/base.py` + `agents/random_agent.py` + `simulation/runner.py` + `simulation/dataset.py`

### Sprint 2: Estrategias (Fases 3-4)
- **Agente A**: `scoring/features.py` + `scoring/scoring_function.py`
- **Agente B**: `agents/greedy_agent.py` + `agents/scorer_agent.py` + `agents/defensive_agent.py` + `agents/offensive_agent.py`
- **Agente C**: `agents/lookahead_agent.py` + `simulation/tournament.py`

### Sprint 3: MCTS + Dataset (Fases 5-6)
- **Agente A**: `agents/mcts_agent.py` (complejo, un agente dedicado)
- **Agente B**: `scripts/generate_dataset.py` + `scripts/run_tournament.py`
- **Agente C**: Tests de integracion: torneo random vs greedy vs scorer, validar dataset

### Sprint 4: Optimizacion + GUI + Analisis (Fases 7-9)
- **Agente A**: `scoring/optimizer.py` + `scripts/optimize_weights.py`
- **Agente B**: `gui/` completo
- **Agente C**: `analysis/` completo + `scripts/explain_strategy.py`

---

## Verificacion

### Como testear end-to-end:
1. `python -m pytest tests/` — todos los unit tests pasan
2. `python scripts/run_tournament.py --agents random,greedy,scorer --games 200` — torneo completo, verificar que scorer > greedy > random
3. `python scripts/generate_dataset.py --games 100 --mcts-iterations 500` — genera dataset sin errores
4. `python scripts/optimize_weights.py --generations 20 --population 15` — pesos mejoran
5. `python scripts/launch_gui.py` — GUI abre, se puede cargar un replay y ver heatmap
6. `python scripts/explain_strategy.py --weights data/weights/optimized.json` — genera consejos legibles

### Metricas de exito:
- MCTS(1000) gana >80% vs Random
- Pesos optimizados ganan >55% vs MCTS(200)
- GUI muestra tablero, fichas, heatmap correctamente
- Explainer genera al menos 5 consejos concretos y accionables
