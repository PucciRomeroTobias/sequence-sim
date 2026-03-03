# Sequence Simulator

A Sequence board game simulator with multiple AI agents, from simple heuristics to MCTS and ML-based approaches.

## Features

- **Full game engine**: 10x10 board, 2-team play, 192 possible lines, card-to-position mapping
- **7 AI agents** of increasing strength: Random, Greedy, Defensive, Offensive, Scorer, Lookahead, SmartAgent, MCTS
- **ML experiments**: Neural network (PyTorch) and LightGBM agents trained on MCTS oracle data
- **Card tracking**: Perfect public information tracking for advanced agents
- **Web interface**: Play against the AI in your browser (React + Socket.IO)
- **Tournament system**: Round-robin tournaments with parallel execution
- **Genetic optimizer**: Evolve scoring weights via tournament-based fitness

## Quick Start

```bash
# Install
pip install -e ".[all]"

# Run tests
python -m pytest tests/ -x -q

# Tournament between agents
python scripts/run_tournament.py --agents random,greedy,defensive,smart --games 100
```

## Web Interface

Play Sequence against the SmartAgent AI in your browser.

### Setup

```bash
# Terminal 1: Start the backend server
pip install aiohttp python-socketio
python web/server.py
# Server starts on http://localhost:8080

# Terminal 2: Start the frontend
cd web/frontend
npm install
npm run dev
# Frontend starts on http://localhost:5173
```

Open http://localhost:5173 in your browser. You play as **blue** (Team 0), the AI plays as **red** (Team 1).

### How to Play

1. Click **New Game** to start
2. Select a card from your hand (bottom of the screen)
3. Click a valid board position to place your chip (highlighted positions are legal moves)
4. The AI will respond after a short delay
5. Complete **2 sequences** of 5-in-a-row to win

**Special cards:**
- **Two-eyed Jacks** (J♦, J♣): Wild — place a chip on any empty position
- **One-eyed Jacks** (J♥, J♠): Remove an opponent's chip from the board
- **Corners**: Free for everyone — count as part of any player's sequences

## Project Structure

```
sequence/
  core/         # Game engine: board, card, deck, game_state, card_tracker, types, actions
  agents/       # AI agents: random, greedy, defensive, lookahead, smart, mcts, neural, lgbm
  scoring/      # 33-feature extraction, scoring function, neural/lgbm scoring, optimizer
  analysis/     # Strategy explainer, statistics
  simulation/   # Game runner, tournament
scripts/        # CLI tools
web/            # Web interface (React + aiohttp + Socket.IO)
tests/          # pytest tests
data/           # Trained models (nn/, lgbm/)
```

## Agents

| Agent | Description | Strength |
|---|---|---|
| `random` | Random legal moves | Baseline |
| `greedy` | Picks the first available sequence-advancing move | Weak |
| `defensive` | Prioritizes blocking opponent threats | Medium |
| `offensive` | Prioritizes advancing own sequences | Medium |
| `scorer` | Linear scoring with balanced weights | Medium |
| `lookahead1` | Depth-1 minimax with scoring function | Medium-Strong |
| `smart` | Card tracking + 33 features + instant decisions + lookahead | Strong |
| `mcts` | Monte Carlo Tree Search with determinization (200 iterations). See [caveat below](#important-caveat-mcts-as-oracle-in-a-hidden-information-game) — suboptimal in hidden-info games | Strong (slow) |
| `hybrid` | LightGBM (MCTS-trained) candidate selection + linear lookahead | Strong |

## ML Experiments

### Important caveat: MCTS as oracle in a hidden-information game

Sequence is a **hidden-information game**: you cannot see your opponent's hand (5-7 cards), and the deck order is unknown. This is a fundamental distinction from perfect-information games like Chess or Go, and it has deep consequences for everything in this section.

MCTS was designed for perfect-information games, where the entire game state is visible to both players. To apply it to Sequence, we use **determinization** (Information Set MCTS): before each search, we randomly sample a plausible opponent hand from the pool of unseen cards, then run standard MCTS on that fully-observable "determinized" state. We repeat this across multiple samples and aggregate visit counts.

This approach has well-documented theoretical limitations:

- **Strategy fusion**: The most fundamental problem. When averaging results across determinizations, MCTS can converge on an action that is "okay on average" but actually optimal in *none* of the individual scenarios. If action A is best when the opponent holds hand X, and action B is best when they hold hand Y, determinized MCTS might pick action C — mediocre in both cases but with more aggregated visits. This is a known failure mode that does not vanish with more iterations.

- **Information leakage during search**: Within each determinized search, MCTS has full visibility of the sampled opponent hand. It explores lines that exploit this knowledge — knowledge the real player does not have. The resulting visit distributions are contaminated by information that should be hidden.

- **No signaling or deception**: Optimal play in hidden-information games sometimes requires signaling (strategically revealing information) or deception (concealing your intentions). Determinized MCTS is structurally blind to these dynamics because it treats each determinization as a separate perfect-information game.

- **No opponent modeling**: MCTS doesn't reason about what the opponent *knows* or can *infer* from your past actions. In a perfect-information game the opponent's best response is deterministic. In a hidden-information game, the opponent's optimal play depends on their beliefs about *your* hand, which determinized MCTS ignores entirely.

- **No convergence guarantee**: In perfect-information games, MCTS provably converges to the minimax-optimal strategy given enough iterations. In hidden-information games with determinization, **this guarantee does not hold**. The algorithm converges to a different solution concept that can be arbitrarily far from Nash equilibrium and is exploitable by an informed opponent.

**Bottom line**: All the ML experiments below use MCTS visit distributions as training signal. Since this "oracle" is itself suboptimal for the reasons above, the ML models are learning to imitate a flawed teacher. Any ceiling on ML performance here is not just about feature engineering or model capacity — it's bounded by the quality of the MCTS signal, which is fundamentally limited by the hidden-information nature of the game.

A truly optimal approach would require algorithms designed for imperfect information (e.g., Counterfactual Regret Minimization, or proper Information Set MCTS variants like ISMCTS with opponent modeling), which are outside the current scope of this project.

### Neural Network (PyTorch)

Trained a small MLP to replace the linear scoring function using MCTS visit distributions as oracle signal. Various architectures and training approaches were tried (pairwise ranking loss, extended features, normalization).

**Result**: High pairwise accuracy (~93% NDCG) but poor gameplay — scores weren't calibrated across positions, breaking lookahead.

```bash
# Generate MCTS oracle data
python scripts/train_neural.py --generate 500 --mcts-iters 1000 \
    --save-data data/nn/training_data.npz

# Train
python scripts/train_neural.py --load-data data/nn/training_data.npz \
    --output data/nn/model.pt --validate
```

### LightGBM LambdaRank

Trained a gradient-boosted tree ranker (LambdaRank objective) to rank actions by distilling MCTS visit distributions. Pure ranking approach without requiring calibrated absolute scores.

**Result**: 48.7% top-1 MCTS agreement (vs SmartAgent's 25.4%), but ranking alone wasn't enough. The **hybrid approach** — LGBM selects candidate actions, linear scoring does tactical lookahead — matches SmartAgent's performance.

```bash
# Train from existing MCTS data
python scripts/train_lgbm.py --data data/nn/training_data_combined.npz \
    --output data/lgbm/model.txt --validate --validation-games 100
```

### Key Finding

With 33 hand-crafted features, SmartAgent's linear scoring + depth-1 lookahead is near-optimal *relative to the MCTS oracle*. ML models learn better action ranking but can't significantly outplay the linear model because:
1. The MCTS oracle itself is suboptimal due to hidden information (see caveat above)
2. The 33 features are a lossy compression of the full game state
3. SmartAgent's lookahead compensates for weaker initial ranking
4. Ranking quality doesn't directly translate to game-winning ability

In other words: we hit the ceiling of what determinized MCTS can teach, not the ceiling of what's achievable in Sequence.

## Scripts

```bash
# Tournament
python scripts/run_tournament.py --agents smart,hybrid,greedy --games 200

# Optimize scoring weights from MCTS data
python scripts/optimize_from_mcts.py --games 200 --mcts-iters 1000

# Strategy report
python scripts/explain_strategy.py --preset smart
```

## Acknowledgments

- **Frontend card assets and game inspiration**: [AnishmMore/sequence-board-game](https://github.com/AnishmMore/sequence-board-game)
- **Frontend stack**: React 19, Vite, Tailwind CSS, Socket.IO
