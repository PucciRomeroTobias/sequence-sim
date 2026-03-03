"""LightGBM LambdaRank scoring function for Sequence game.

Learns to rank actions by distilling MCTS visit distributions into a fast
gradient-boosted tree model. Uses LambdaRank objective (optimizes NDCG)
which is designed for ranking within groups (positions).

Unlike the neural network approach, this does NOT need calibrated absolute
scores across positions — it only ranks actions within each position.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .features import NUM_FEATURES, extract_features

if TYPE_CHECKING:
    from ..core.actions import Action
    from ..core.card_tracker import CardTracker
    from ..core.game_state import GameState
    from ..core.types import TeamId


def _import_lightgbm():
    """Lazy import lightgbm to avoid hard dependency."""
    try:
        import lightgbm as lgb
        return lgb
    except ImportError as e:
        raise ImportError(
            "LightGBM is required for LGBM scoring. "
            "Install with: pip install lightgbm"
        ) from e


class LGBMScoringFunction:
    """Drop-in scoring function using a trained LightGBM ranker.

    Interface:
    - rank_actions(state, legal_actions, team, tracker=None) -> list[(Action, float)]
    """

    def __init__(self, model_path: str) -> None:
        lgb = _import_lightgbm()
        self._model = lgb.Booster(model_file=model_path)

    def rank_actions(
        self,
        state: GameState,
        legal_actions: list[Action],
        team: TeamId,
        tracker: CardTracker | None = None,
    ) -> list[tuple[Action, float]]:
        """Rank actions using batched LightGBM prediction with virtual apply.

        For PLACE actions (~90%), mutates board/hand in-place temporarily
        to avoid expensive deep copies. Collects all feature vectors into
        a single (N, 33) array for one batched prediction.
        """
        from ..core.actions import ActionType
        from ..core.types import EMPTY

        board = state.board
        chips = board.chips
        hand = state.hands.get(team.value, [])

        features_list: list[np.ndarray] = []
        action_indices: list[int] = []

        for i, action in enumerate(legal_actions):
            if action.action_type == ActionType.PLACE and action.position is not None:
                pos = action.position
                r, c = pos.row, pos.col
                # --- Virtual apply ---
                old_chip = int(chips[r, c])
                chips[r, c] = team.value
                board.empty_positions.discard(pos)
                hand.remove(action.card)
                state.turn_number += 1

                features = extract_features(state, team, tracker=tracker)
                features_list.append(features)
                action_indices.append(i)

                # --- Revert ---
                state.turn_number -= 1
                hand.append(action.card)
                chips[r, c] = old_chip
                if old_chip == EMPTY:
                    board.empty_positions.add(pos)
            else:
                next_state = state.apply_action(action)
                features = extract_features(next_state, team, tracker=tracker)
                features_list.append(features)
                action_indices.append(i)

        if not features_list:
            return [(a, 0.0) for a in legal_actions]

        batch = np.array(features_list, dtype=np.float64)
        scores = self._model.predict(batch)

        scored: list[tuple[Action, float]] = []
        for j, idx in enumerate(action_indices):
            scored.append((legal_actions[idx], float(scores[j])))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored


def prepare_lambdarank_data(
    data_path: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert MCTS data (.npz) to LambdaRank format.

    Returns:
        X: (total_actions, 33) feature matrix
        y: (total_actions,) relevance labels (visit proportions, 0 to 1)
        groups: (n_positions,) number of actions per group/position
    """
    data = np.load(data_path, allow_pickle=True)
    lengths = data["lengths"]

    all_features: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    groups: list[int] = []

    for j in range(len(lengths)):
        features = data[f"f_{j}"]  # (n_actions, 33)
        visits = data[f"v_{j}"]    # (n_actions,)

        n_actions = len(visits)
        if n_actions < 2:
            continue

        # Scale visit proportions to integer relevance labels (0 to 100)
        # Preserves magnitude of differences: 50% vs 5% is much bigger
        # than 50% vs 40%, which ordinal ranks would lose.
        # LambdaRank requires integer labels.
        total_visits = visits.sum()
        if total_visits <= 0:
            continue
        labels = np.round(100.0 * visits / total_visits).astype(np.int32)

        all_features.append(features)
        all_labels.append(labels)
        groups.append(n_actions)

    X = np.concatenate(all_features, axis=0).astype(np.float64)
    y = np.concatenate(all_labels, axis=0).astype(np.float64)
    groups_arr = np.array(groups, dtype=np.int32)

    return X, y, groups_arr


def train_lgbm_ranker(
    data_path: str,
    num_leaves: int = 31,
    n_estimators: int = 300,
    learning_rate: float = 0.05,
    min_child_samples: int = 20,
    val_fraction: float = 0.2,
    early_stopping_rounds: int = 30,
    verbose: bool = True,
) -> object:
    """Train a LightGBM LambdaRank model from MCTS data.

    Args:
        data_path: Path to .npz file with MCTS training data.
        num_leaves: Max leaves per tree.
        n_estimators: Max number of boosting rounds.
        learning_rate: Shrinkage rate.
        min_child_samples: Min samples per leaf.
        val_fraction: Fraction of groups for validation.
        early_stopping_rounds: Stop if val metric doesn't improve.
        verbose: Print progress.

    Returns:
        Trained LightGBM Booster.
    """
    lgb = _import_lightgbm()

    X, y, groups = prepare_lambdarank_data(data_path)

    if verbose:
        print(f"Data: {X.shape[0]} actions, {len(groups)} positions, {X.shape[1]} features")
        print(f"Label distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    # Train/val split by groups (keep groups intact)
    n_groups = len(groups)
    n_val_groups = max(1, int(n_groups * val_fraction))
    n_train_groups = n_groups - n_val_groups

    # Shuffle groups
    rng = np.random.default_rng(42)
    group_indices = rng.permutation(n_groups)

    train_group_idx = np.sort(group_indices[:n_train_groups])
    val_group_idx = np.sort(group_indices[n_train_groups:])

    # Build cumulative index for slicing
    cum_groups = np.concatenate([[0], np.cumsum(groups)])

    train_rows = np.concatenate([
        np.arange(cum_groups[i], cum_groups[i + 1]) for i in train_group_idx
    ])
    val_rows = np.concatenate([
        np.arange(cum_groups[i], cum_groups[i + 1]) for i in val_group_idx
    ])

    X_train, y_train = X[train_rows], y[train_rows]
    X_val, y_val = X[val_rows], y[val_rows]
    train_groups = groups[train_group_idx]
    val_groups = groups[val_group_idx]

    if verbose:
        print(f"Train: {len(train_groups)} groups, {len(X_train)} actions")
        print(f"Val: {len(val_groups)} groups, {len(X_val)} actions")

    train_set = lgb.Dataset(X_train, label=y_train, group=train_groups)
    val_set = lgb.Dataset(X_val, label=y_val, group=val_groups, reference=train_set)

    # label_gain maps integer labels to NDCG gains; we use 0-100 scale
    label_gain = ",".join(str(i) for i in range(101))

    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "eval_at": [1, 3, 5],
        "label_gain": label_gain,
        "num_leaves": num_leaves,
        "learning_rate": learning_rate,
        "min_child_samples": min_child_samples,
        "verbose": -1,
        "seed": 42,
    }

    callbacks = []
    if early_stopping_rounds > 0:
        callbacks.append(lgb.early_stopping(early_stopping_rounds, verbose=verbose))
    if verbose:
        callbacks.append(lgb.log_evaluation(period=20))

    model = lgb.train(
        params,
        train_set,
        num_boost_round=n_estimators,
        valid_sets=[val_set],
        valid_names=["val"],
        callbacks=callbacks,
    )

    if verbose:
        best_iter = model.best_iteration
        print(f"\nBest iteration: {best_iter}")

        # Feature importance
        importance = model.feature_importance(importance_type="gain")
        feature_names = [f"f_{i}" for i in range(X.shape[1])]
        sorted_idx = np.argsort(-importance)
        print("\nTop 10 features by gain:")
        for idx in sorted_idx[:10]:
            print(f"  {feature_names[idx]} (idx={idx}): {importance[idx]:.1f}")

    return model
