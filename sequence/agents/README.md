# AI Agents

## Overview

All agents implement the `Agent` base class (`base.py`):
- `choose_action(state, legal_actions) -> Action`
- `notify_game_start(team, config)`
- `notify_action(action, team)`

## Agent Hierarchy

| Agent | Win Rate vs Greedy | Key Idea |
|---|---|---|
| RandomAgent | ~25% | Uniform random legal moves |
| GreedyAgent | baseline | First move that advances a sequence |
| DefensiveAgent | ~50% | Prioritizes blocking opponent threats |
| OffensiveAgent | ~45% | Prioritizes advancing own lines |
| ScorerAgent | ~50% | Linear scoring with balanced weights |
| LookaheadAgent | ~55% | Depth-1/2 minimax with scoring |
| **SmartAgent** | **~50%** | Card tracking + 33 features + instant decisions + lookahead |
| MCTSAgent | ~55% (slow) | Monte Carlo Tree Search, 200-1000 iterations |
| NeuralAgent | ~20% | MLP trained on MCTS data (failed experiment) |
| LGBMAgent | ~42% | LightGBM LambdaRank (failed as standalone) |
| **HybridAgent** | **~50%** | LGBM candidates + linear lookahead (matches SmartAgent) |

---

## SmartAgent

**File**: `smart_agent.py`

The strongest heuristic agent. Combines:

1. **CardTracker**: Perfect public information tracking — knows which cards have been played, how many copies of each card remain, which positions are permanently dead
2. **33-feature evaluation**: Hand-crafted features covering offensive lines, defensive threats, card availability, spatial control, and game phase
3. **Instant decisions**: Deterministic shortcuts for obvious moves — completing a sequence, blocking opponent's 4-in-a-row, creating forks. These skip the scoring pipeline entirely
4. **Dead card boost**: Incentivizes discarding dead cards (cards whose board positions are all occupied) when the score difference is small
5. **Depth-1 lookahead**: For top 5 candidates, simulates our move + opponent's best response, picks the move with the best worst-case outcome

### Strengths
- Robust: simple linear model doesn't overfit
- Instant decisions catch critical tactical moments
- Lookahead compensates for imperfect feature-based evaluation
- Fast enough for real-time play

### Limitations
- Linear scoring can't capture feature interactions (e.g., "fork + opponent has no one-eyed-jack" is much better than fork alone)
- Depth-1 lookahead is shallow — misses multi-step tactics
- Hand-tuned weights (SMART_WEIGHTS) may not be optimal; genetic optimization showed marginal improvements

---

## NeuralAgent

**Files**: `neural_agent.py`, `../scoring/neural_scoring.py`

An attempt to replace SmartAgent's linear `dot(weights, features)` with a small MLP (multi-layer perceptron) that could learn non-linear feature interactions.

### Architecture
- `ActionValueNet`: Configurable MLP (default: 2 hidden layers, 128 units, ReLU)
- Input: 33 features (same as SmartAgent)
- Output: scalar score per action
- Batched evaluation with virtual-apply optimization

### Training
- **Data**: MCTS oracle — 5,525 positions from 750 games (MCTS 1000 iterations with informed determinization vs SmartAgent)
- **Loss**: Pairwise ranking loss `log(1 + exp(-(score_better - score_worse)))` using MCTS visit counts as labels
- **Validation**: 80/20 train/val split on pairwise accuracy

### Results

| Model Variant | Val Pairwise Accuracy | Win Rate vs SmartAgent |
|---|---|---|
| h128, 2 layers | 70.2% | 45% |
| h256, 2 layers | 70.5% | 44% |
| 3 layers | 67.8% | — |
| Extended (48 features) | 79.4% | 31.5% |
| Normalized (z-score) | 91.3% | 32% |

### Why It Failed

**Higher pairwise accuracy did NOT translate to better gameplay.** The root cause:

1. **Pairwise ranking ≠ absolute value**: The model learned to rank actions *within* a position (which MCTS prefers), but produced uncalibrated scores *across* positions. A score of 5.0 in one game state meant something completely different than 5.0 in another
2. **Lookahead requires calibrated scores**: SmartAgent's lookahead compares scores from different future states. With uncalibrated neural scores, this comparison was meaningless, so lookahead actually hurt performance
3. **Dead card boost broke**: This heuristic adds +2.0 to dead card discard scores when close to the best score. With uncalibrated neural scores, this threshold was meaningless

The normalized model (91.3% accuracy) was the extreme case: z-score normalization made pairwise accuracy excellent but compressed the score range so much that all cross-state comparisons became noise.

### What Would Be Needed
- Much more training data (50k+ positions, not 5.5k)
- A different training objective (game outcome prediction instead of pairwise ranking)
- Or using the NN inside MCTS (AlphaZero-style) where cross-state calibration doesn't matter

---

## LGBMAgent / HybridAgent

**Files**: `lgbm_agent.py`, `../scoring/lgbm_scoring.py`

### LGBMAgent (Pure Ranking)

Gradient-boosted tree ranker trained with LightGBM's LambdaRank objective (optimizes NDCG). Same idea as the neural approach but with a model better suited for tabular data.

**Training**: Same MCTS oracle data, visit counts converted to 0-100 integer relevance labels grouped by position.

**Result**: 48.7% top-1 MCTS agreement (vs SmartAgent's 25.4% and neural's 44.4%). The model is almost twice as good as SmartAgent at picking the move MCTS would choose. But as a standalone agent, it loses to SmartAgent (38% vs Smart, 42% vs Greedy).

**Why**: Same fundamental problem — ranking within a position is not enough. SmartAgent's depth-1 lookahead catches tactical errors that any single-shot model misses, regardless of how good its ranking is.

### HybridAgent (Best ML Approach)

Combines the best of both worlds:
1. **LGBM** ranks all legal actions, selects top 5 candidates (better candidate selection than SmartAgent's linear scoring)
2. **SmartAgent's linear scoring** does depth-1 lookahead on those 5 candidates (calibrated cross-state comparison)

**Result**: Matches SmartAgent performance (~50% head-to-head, ~56% vs Greedy). This is the first ML approach that equaled the heuristic baseline.

### Key Findings

1. **Feature bottleneck**: The 33 features are a lossy compression of the full game state. MCTS reasons about specific card combinations, multi-step sequences, and opponent hand probabilities that these features can't capture
2. **SmartAgent is near-optimal for 33 features**: With these features, simple linear scoring + lookahead is already close to the best achievable. ML adds marginal ranking improvement but can't overcome the information loss
3. **Ranking ≠ playing**: A model can perfectly predict which move MCTS prefers and still lose games if it can't evaluate positions absolutely

---

## The Data Problem

All ML experiments suffer from insufficient and biased training data:

1. **Only 5,525 positions** from 750 MCTS games — a tiny dataset for ML. Each game takes ~30 seconds with MCTS at 1000 iterations, so generating 50k+ positions would take hours
2. **Biased sampling**: We only collected data from positions where MCTS had multiple choices AND the top visit count was >10% of total. This means the model only trains on "interesting" ambiguous positions, never seeing the many straightforward positions that make up most of a game
3. **Single opponent**: All data comes from MCTS vs SmartAgent. The model never sees play patterns from defensive or aggressive styles
4. **No real game data**: All training data is synthetic (AI vs AI). Real human games would provide much richer and more diverse positions, especially in the opening and endgame phases where human and AI strategies diverge significantly

### What Would Actually Help
- **10x more MCTS data** with diverse opponents and game phases
- **Real game data** from online Sequence platforms, if available
- **Richer features** or raw board representation (10x10 grid + hand + game state) instead of 33 hand-crafted features
- **Game outcome signal** instead of move-level ranking: train the model to predict win probability from a position, which naturally produces calibrated scores
- **Self-play iteration**: Train a model, play it against itself, retrain on the new games, repeat (AlphaZero approach)
