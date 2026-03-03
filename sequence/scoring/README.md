# Scoring System

## Feature Extraction (`features.py`)

33 hand-crafted features extracted from a `GameState` for a given team. These are the input to all scoring approaches (linear, neural, LightGBM).

### Feature List

| # | Feature | Description |
|---|---|---|
| 0 | completed_sequences | Own completed 5-in-a-row sequences |
| 1 | four_in_a_row | Own lines with 4 chips (one away from sequence) |
| 2 | three_in_a_row | Own lines with 3 chips |
| 3 | two_in_a_row | Own lines with 2 chips |
| 4 | opp_completed_sequences | Opponent's completed sequences |
| 5 | opp_four_in_a_row | Opponent lines with 4 chips |
| 6 | opp_three_in_a_row | Opponent lines with 3 chips |
| 7 | chips_on_board | Total own chips placed |
| 8 | opp_chips_on_board | Total opponent chips placed |
| 9 | center_control | Own chips in rows/cols 2-7 |
| 10 | corner_adjacency | Own chips adjacent to corner positions |
| 11 | hand_pairs | Duplicate cards in hand |
| 12 | two_eyed_jacks | Two-eyed jacks in hand (wild placement) |
| 13 | one_eyed_jacks | One-eyed jacks in hand (removal) |
| 14 | dead_cards | Cards in hand with no empty board positions |
| 15 | shared_line_potential | Lines with own chips and no opponent |
| 16 | blocked_lines | Lines with both own and opponent chips |
| **17-27** | **Card-tracking features** | **Require CardTracker (default 0 without)** |
| 17 | guaranteed_positions | Positions only we can fill (monopoly on remaining cards) |
| 18 | viable_four_in_a_row | 4-in-a-row where the gap can actually be filled |
| 19 | viable_three_in_a_row | 3-in-a-row where all gaps can be filled |
| 20 | opp_blockable_four | Opponent 4-in-a-row we can block |
| 21 | opp_unblockable_four | Opponent 4-in-a-row we cannot block |
| 22 | fork_count | Positions advancing 2+ lines with 3+ own chips |
| 23 | dead_positions | Empty positions permanently unfillable |
| 24 | dead_lines | Own lines with permanently dead positions |
| 25 | opp_dead_lines | Opponent lines with permanently dead positions |
| 26 | card_monopoly | Cards where we hold all remaining copies |
| 27 | sequence_proximity | Weighted sum of progress toward sequences |
| **28-32** | **Advanced strategy features** | |
| 28 | position_line_score | Sum of line memberships for own chips |
| 29 | anchor_overlap | Completed sequence chips that feed new lines |
| 30 | chip_clustering | Max chip concentration in a 5x5 quadrant |
| 31 | early_jack_penalty | Penalty for using jacks early (preserves flexibility) |
| 32 | jack_save_value | Value of saving jacks for later (decays over time) |

### Design Decisions

- **CardTracker dependency**: Features 17-27 require a CardTracker for card availability analysis. Without it they default to 0, making the feature set backward-compatible with simpler agents
- **Virtual apply**: During action ranking, features are extracted after "virtually applying" each action (mutate board in-place, extract, revert) to avoid expensive deep copies. This is ~5x faster than `state.apply_action()`
- **No raw board**: We deliberately use hand-crafted features rather than the raw 10x10 board. This keeps the feature space small (33 dimensions) but loses information about specific card positions and combinatorial interactions

### Known Limitations

The 33 features capture game state at a high level but lose important information:

- **Specific card positions**: The model knows "3 chips in a line" but not *which* specific line or positions
- **Multi-step tactics**: Features describe the current state, not what happens 2-3 moves ahead
- **Opponent hand inference**: CardTracker tracks what's been played, but doesn't model what the opponent likely holds
- **Interaction effects**: "Fork + opponent lacks one-eyed-jack" is exponentially more dangerous than either alone, but linear scoring treats them independently

## Scoring Approaches

### Linear (`scoring_function.py`)

`score = dot(weights, features)` — used by SmartAgent and ScorerAgent.

- **SMART_WEIGHTS**: 33 hand-tuned weights with strong priors (completed_sequences=91, opp_completed=-100, etc.)
- **ScoringWeights**: Backward-compatible dataclass that handles arrays of different lengths
- **rank_actions_fast()**: Batched evaluation with virtual apply optimization

### Neural (`neural_scoring.py`)

MLP replacement for the linear scoring. See `../agents/README.md` for experiment details.

- **ActionValueNet**: Configurable MLP (hidden size, layers, extended features)
- **NeuralScoringFunction**: Drop-in compatible with ScoringFunction interface
- Supports normalization stats (feat_mean/feat_std) stored in checkpoint
- **train_model()**: Pairwise ranking loss with early stopping

### LightGBM (`lgbm_scoring.py`)

Gradient-boosted tree ranker. See `../agents/README.md` for experiment details.

- **LGBMScoringFunction**: Loads a trained LightGBM Booster model
- **prepare_lambdarank_data()**: Converts MCTS .npz data to LambdaRank format (features, relevance labels, groups)
- **train_lgbm_ranker()**: Trains with LambdaRank objective, NDCG metric, early stopping

## Training Data Format

MCTS oracle data is stored as `.npz` files with:
- `lengths`: array of action counts per position
- `f_{i}`: feature matrix (n_actions, 33) for position i
- `v_{i}`: MCTS visit counts (n_actions,) for position i

Current dataset: 5,525 positions (~60k action data points) from 750 MCTS games. This is small by ML standards — the features simply don't vary enough across these positions for a model to learn patterns significantly better than the hand-tuned linear weights.
