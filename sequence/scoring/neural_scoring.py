"""Neural network scoring function for evaluating Sequence game states.

Replaces the linear dot(weights, features) with a small MLP that learns
non-linear feature interactions (forks + jack absence, multi-line attacks, etc.).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .features import (
    NUM_EXTENDED_FEATURES,
    NUM_FEATURES,
    extract_features,
    extract_features_extended,
)

if TYPE_CHECKING:
    from ..core.actions import Action
    from ..core.card_tracker import CardTracker
    from ..core.game_state import GameState
    from ..core.types import TeamId


def _import_torch():
    """Lazy import torch to avoid hard dependency."""
    try:
        import torch
        return torch
    except ImportError as e:
        raise ImportError(
            "PyTorch is required for neural scoring. "
            "Install with: pip install -e '.[nn]'"
        ) from e


class ActionValueNet:
    """Small MLP for state evaluation: (batch, 33) -> (batch,).

    Architecture is configurable via `num_layers` (default 2 hidden layers).
    """

    def __init__(
        self, hidden: int = 128, num_layers: int = 2, extended: bool = False,
    ) -> None:
        torch = _import_torch()
        nn = torch.nn

        input_size = NUM_EXTENDED_FEATURES if extended else NUM_FEATURES
        layers: list = []
        in_size = input_size
        for _ in range(num_layers):
            layers.append(nn.Linear(in_size, hidden))
            layers.append(nn.ReLU())
            in_size = hidden
        layers.append(nn.Linear(hidden, 1))

        self._model = nn.Sequential(*layers)

    @property
    def model(self):
        return self._model

    def forward(self, x):
        """Forward pass: (batch, 33) -> (batch,)."""
        return self._model(x).squeeze(-1)

    def parameters(self):
        return self._model.parameters()

    def train(self):
        self._model.train()

    def eval(self):
        self._model.eval()

    def state_dict(self):
        return self._model.state_dict()

    def load_state_dict(self, state_dict):
        self._model.load_state_dict(state_dict)

    def to(self, device):
        self._model.to(device)
        return self


class NeuralScoringFunction:
    """Drop-in replacement for ScoringFunction that uses a neural network.

    Compatible interface:
    - evaluate(state, team, tracker=None) -> float
    - rank_actions_fast(state, legal_actions, team, tracker=None) -> list[(Action, float)]
    """

    def __init__(
        self, model_path: str, hidden: int = 128, num_layers: int = 2,
        extended: bool = False,
    ) -> None:
        torch = _import_torch()
        self._torch = torch
        self._extended = extended
        self._feat_mean: np.ndarray | None = None
        self._feat_std: np.ndarray | None = None

        self._net = ActionValueNet(hidden=hidden, num_layers=num_layers, extended=extended)
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

        # Support both formats: raw state_dict or dict with normalization stats
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            self._net.load_state_dict(checkpoint["model"])
            if "feat_mean" in checkpoint:
                self._feat_mean = np.array(checkpoint["feat_mean"], dtype=np.float32)
                self._feat_std = np.array(checkpoint["feat_std"], dtype=np.float32)
        else:
            self._net.load_state_dict(checkpoint)
        self._net.eval()

    def evaluate(
        self,
        state: GameState,
        team: TeamId,
        tracker: CardTracker | None = None,
    ) -> float:
        """Compute a scalar score for the given team's position."""
        torch = self._torch
        extract = extract_features_extended if self._extended else extract_features
        features = extract(state, team, tracker=tracker).astype(np.float32)
        if self._feat_mean is not None:
            features = (features - self._feat_mean) / self._feat_std
        with torch.no_grad():
            tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            score = self._net.forward(tensor)
        return float(score.item())

    def rank_actions_fast(
        self,
        state: GameState,
        legal_actions: list[Action],
        team: TeamId,
        tracker: CardTracker | None = None,
    ) -> list[tuple[Action, float]]:
        """Rank actions using batched neural evaluation with virtual apply.

        For PLACE actions (~90%), mutates board/hand in-place temporarily.
        Accumulates all feature vectors into a single (N, 33) tensor for
        one batched forward pass instead of N individual evaluations.
        """
        from ..core.actions import ActionType
        from ..core.types import EMPTY

        torch = self._torch
        board = state.board
        chips = board.chips
        hand = state.hands.get(team.value, [])
        team_val = team.value

        extract = extract_features_extended if self._extended else extract_features
        features_list: list[np.ndarray] = []
        action_indices: list[int] = []

        for i, action in enumerate(legal_actions):
            if action.action_type == ActionType.PLACE and action.position is not None:
                pos = action.position
                r, c = pos.row, pos.col
                # --- Virtual apply: mutate in-place ---
                old_chip = int(chips[r, c])
                chips[r, c] = team_val
                board.empty_positions.discard(pos)
                hand.remove(action.card)
                state.turn_number += 1

                features = extract(state, team, tracker=tracker)
                features_list.append(features)
                action_indices.append(i)

                # --- Revert ---
                state.turn_number -= 1
                hand.append(action.card)
                chips[r, c] = old_chip
                if old_chip == EMPTY:
                    board.empty_positions.add(pos)
            else:
                # Fallback for REMOVE and DEAD_CARD_DISCARD
                next_state = state.apply_action(action)
                features = extract(next_state, team, tracker=tracker)
                features_list.append(features)
                action_indices.append(i)

        # Batched forward pass
        if features_list:
            batch_np = np.array(features_list, dtype=np.float32)
            if self._feat_mean is not None:
                batch_np = (batch_np - self._feat_mean) / self._feat_std
            with torch.no_grad():
                batch = torch.tensor(batch_np, dtype=torch.float32)
                scores = self._net.forward(batch).numpy()
        else:
            scores = np.array([])

        # Build result
        scored: list[tuple[Action, float]] = []
        for j, idx in enumerate(action_indices):
            scored.append((legal_actions[idx], float(scores[j])))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored


def _extend_features_batch(base: np.ndarray) -> np.ndarray:
    """Compute 15 derived features from 33 base features.

    Works on a batch: (N, 33) -> (N, 48).
    Mirrors the logic in extract_features_extended() but operates on
    pre-extracted feature arrays without needing GameState.

    Note: game_phase (feature 46) is approximated from chips_on_board
    since we don't have turn_number in the stored data.
    """
    # Base feature indices
    four_row = base[:, 1]
    three_row = base[:, 2]
    opp_four = base[:, 5]
    chips = base[:, 7]
    opp_chips = base[:, 8]
    center = base[:, 9]
    one_eyed_jacks = base[:, 13]
    dead_cards = base[:, 14]
    shared_line = base[:, 15]
    guaranteed = base[:, 17]
    viable_four = base[:, 18]
    viable_three = base[:, 19]
    opp_blockable = base[:, 20]
    fork = base[:, 22]
    monopoly = base[:, 26]
    seq_proximity = base[:, 27]
    pos_line_score = base[:, 28]
    anchor_overlap = base[:, 29]
    clustering = base[:, 30]
    jack_save = base[:, 32]

    # Approximate game phase from total chips placed
    # Each player places ~1 chip per turn, so total chips ≈ turn_number.
    # Live code uses turn_number / 80.0, so we use the same denominator.
    game_phase = np.clip((chips + opp_chips) / 80.0, 0.0, 1.0)

    derived = np.column_stack([
        four_row * viable_four,                         # 33
        three_row * viable_three,                       # 34
        opp_four * one_eyed_jacks,                      # 35
        opp_four * (1.0 / (1.0 + one_eyed_jacks)),     # 36
        fork * (1.0 / (1.0 + opp_blockable)),           # 37
        guaranteed * seq_proximity,                      # 38
        four_row * opp_four,                             # 39
        clustering * pos_line_score,                     # 40
        center * shared_line,                            # 41
        dead_cards * game_phase,                         # 42
        jack_save * fork * game_phase,                   # 43
        monopoly * viable_three,                         # 44
        anchor_overlap * three_row,                      # 45
        game_phase,                                      # 46
        chips - opp_chips,                               # 47
    ])

    return np.concatenate([base, derived], axis=1)


def train_model(
    data_path: str,
    hidden: int = 128,
    num_layers: int = 2,
    extended: bool = False,
    epochs: int = 100,
    lr: float = 0.001,
    batch_size: int = 256,
    patience: int = 10,
) -> ActionValueNet:
    """Train an ActionValueNet using pairwise ranking loss from MCTS data.

    Args:
        data_path: Path to .npz file with features_{i} and visits_{i} arrays.
        hidden: Hidden layer size.
        epochs: Max training epochs.
        lr: Learning rate.
        batch_size: Training batch size.
        patience: Early stopping patience (epochs without val loss improvement).

    Returns:
        Best model (by validation loss).
    """
    torch = _import_torch()

    # --- Load data ---
    data = np.load(data_path, allow_pickle=True)
    lengths = data["lengths"]
    samples: list[dict] = []
    for j in range(len(lengths)):
        samples.append({
            "features": data[f"f_{j}"],
            "visits": data[f"v_{j}"],
        })

    print(f"Loaded {len(samples)} positions from {data_path}")

    # --- Build pairwise data ---
    better_features: list[np.ndarray] = []
    worse_features: list[np.ndarray] = []

    for sample in samples:
        features = sample["features"]  # (n_actions, 33)
        if extended and features.shape[1] == NUM_FEATURES:
            features = _extend_features_batch(features)
        visits = sample["visits"]      # (n_actions,)

        best_idx = int(np.argmax(visits))
        f_best = features[best_idx]

        sorted_indices = np.argsort(-visits)
        for idx in sorted_indices[1:6]:  # top 5 pairs
            if visits[idx] < visits[best_idx]:
                better_features.append(f_best)
                worse_features.append(features[idx])

    if not better_features:
        raise ValueError("No pairwise data generated from the dataset")

    better_arr = np.array(better_features, dtype=np.float32)
    worse_arr = np.array(worse_features, dtype=np.float32)
    n_pairs = len(better_arr)
    print(f"Built {n_pairs} pairwise training samples")

    # --- Train/val split ---
    indices = np.random.permutation(n_pairs)
    split = int(0.8 * n_pairs)
    train_idx = indices[:split]
    val_idx = indices[split:]

    train_better = torch.tensor(better_arr[train_idx])
    train_worse = torch.tensor(worse_arr[train_idx])
    val_better = torch.tensor(better_arr[val_idx])
    val_worse = torch.tensor(worse_arr[val_idx])

    print(f"Train: {len(train_idx)} pairs, Val: {len(val_idx)} pairs")

    # --- Model and optimizer ---
    net = ActionValueNet(hidden=hidden, num_layers=num_layers, extended=extended)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    best_val_loss = float("inf")
    best_state_dict = None
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        net.train()

        # Shuffle training data
        perm = torch.randperm(len(train_better))
        train_better_shuffled = train_better[perm]
        train_worse_shuffled = train_worse[perm]

        total_loss = 0.0
        n_batches = 0

        for start in range(0, len(train_better_shuffled), batch_size):
            end = start + batch_size
            b_better = train_better_shuffled[start:end]
            b_worse = train_worse_shuffled[start:end]

            score_better = net.forward(b_better)
            score_worse = net.forward(b_worse)

            # Pairwise ranking loss: log(1 + exp(-(s_better - s_worse)))
            diff = score_better - score_worse
            loss = torch.log1p(torch.exp(-diff)).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        train_loss = total_loss / max(n_batches, 1)

        # --- Validation ---
        net.eval()
        with torch.no_grad():
            val_score_better = net.forward(val_better)
            val_score_worse = net.forward(val_worse)
            val_diff = val_score_better - val_score_worse
            val_loss = torch.log1p(torch.exp(-val_diff)).mean().item()
            val_acc = (val_diff > 0).float().mean().item()

            train_score_better = net.forward(train_better)
            train_score_worse = net.forward(train_worse)
            train_diff = train_score_better - train_score_worse
            train_acc = (train_diff > 0).float().mean().item()

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:>3}/{epochs} | "
                f"Train loss: {train_loss:.4f} acc: {train_acc:.1%} | "
                f"Val loss: {val_loss:.4f} acc: {val_acc:.1%}"
            )

        # Early stopping
        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_state_dict = {k: v.clone() for k, v in net.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch} (patience={patience})")
                break

    # Restore best model
    if best_state_dict is not None:
        net.load_state_dict(best_state_dict)

    # Final metrics
    net.eval()
    with torch.no_grad():
        val_score_better = net.forward(val_better)
        val_score_worse = net.forward(val_worse)
        val_diff = val_score_better - val_score_worse
        final_val_acc = (val_diff > 0).float().mean().item()
    print(f"\nBest model: val loss={best_val_loss:.4f}, val acc={final_val_acc:.1%}")

    return net
