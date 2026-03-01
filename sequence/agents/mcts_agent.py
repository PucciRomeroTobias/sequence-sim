"""MCTS agent with Information Set determinization for imperfect information."""

from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING

from ..core.board import ALL_LINES, CARD_TO_POSITIONS
from ..core.types import CORNER, TeamId
from .base import Agent

if TYPE_CHECKING:
    from ..core.actions import Action
    from ..core.game import GameConfig
    from ..core.game_state import GameState

# Pre-compute TeamId instances to avoid repeated enum construction
_TEAM_IDS = [TeamId(0), TeamId(1), TeamId(2)]


class MCTSNode:
    """A node in the MCTS search tree."""

    __slots__ = (
        "state",
        "parent",
        "action",
        "children",
        "untried_actions",
        "visits",
        "total_value",
    )

    def __init__(
        self,
        state: GameState,
        parent: MCTSNode | None = None,
        action: Action | None = None,
        legal_actions: list[Action] | None = None,
    ) -> None:
        self.state = state
        self.parent = parent
        self.action = action
        self.children: dict[Action, MCTSNode] = {}
        self.untried_actions: list[Action] = (
            legal_actions
            if legal_actions is not None
            else state.get_legal_actions()
        )
        self.visits: int = 0
        self.total_value: float = 0.0

    def ucb1(self, exploration_constant: float) -> float:
        if self.visits == 0:
            return float("inf")
        return (
            self.total_value / self.visits
            + exploration_constant * math.sqrt(math.log(self.parent.visits) / self.visits)
        )

    def best_child(self, c: float) -> MCTSNode:
        return max(self.children.values(), key=lambda n: n.ucb1(c))

    def is_fully_expanded(self) -> bool:
        return len(self.untried_actions) == 0


def _fast_board_eval(chips, team_val: int, num_teams: int) -> float:
    """Fast board evaluation using numpy for the rollout heuristic.

    Returns value in [0, 1] range.
    """
    score = 0.0
    for line in ALL_LINES:
        own = 0
        opp = 0
        for pos in line:
            v = int(chips[pos.row, pos.col])
            if v == team_val or v == CORNER:
                own += 1
            elif v >= 0 and v != CORNER and v != -1:
                opp += 1
        if opp == 0:
            if own == 5:
                return 1.0  # We completed a sequence
            elif own == 4:
                score += 50
            elif own == 3:
                score += 5
        if own == 0:
            if opp == 5:
                return 0.0  # Opponent completed
            elif opp == 4:
                score -= 40
            elif opp == 3:
                score -= 4
    # Normalize to [0, 1] with sigmoid
    return 1.0 / (1.0 + math.exp(-score / 100.0))


class MCTSAgent(Agent):
    """Monte Carlo Tree Search agent with Information Set determinization.

    Handles imperfect information by sampling plausible opponent hands
    and running MCTS on each determinized state.
    """

    def __init__(
        self,
        iterations: int = 1000,
        num_determinizations: int = 10,
        exploration_constant: float = 1.41,
        rollout_depth: int = 30,
        scoring_fn: object | None = None,
        max_root_actions: int = 20,
        seed: int | None = None,
    ) -> None:
        self._iterations = iterations
        self._num_determinizations = num_determinizations
        self._exploration_constant = exploration_constant
        self._rollout_depth = rollout_depth
        self._scoring_fn = scoring_fn
        self._max_root_actions = max_root_actions
        self._rng = random.Random(seed)
        self._team: TeamId | None = None

        # Stored after each decision for dataset generation
        self.last_mcts_visits: dict[str, int] | None = None
        self.last_mcts_values: dict[str, float] | None = None

    def notify_game_start(self, team: TeamId, config: GameConfig) -> None:
        self._team = team

    def notify_action(self, action: Action, team: TeamId) -> None:
        pass

    def choose_action(
        self, state: GameState, legal_actions: list[Action]
    ) -> Action:
        if self._team is None:
            self._team = state.current_team

        if len(legal_actions) == 1:
            self.last_mcts_visits = {str(legal_actions[0]): 1}
            self.last_mcts_values = {str(legal_actions[0]): 0.0}
            return legal_actions[0]

        # Pre-filter if too many actions
        filtered = self._filter_root_actions(state, legal_actions)

        # Build action-to-index map for fast lookup
        action_to_idx = {a: i for i, a in enumerate(filtered)}

        # Aggregate visit counts across determinizations
        total_visits = [0] * len(filtered)
        total_values = [0.0] * len(filtered)

        iters_per_det = max(1, self._iterations // self._num_determinizations)

        for _ in range(self._num_determinizations):
            det_state = self._determinize(state)
            root = MCTSNode(det_state, legal_actions=list(filtered))

            for _ in range(iters_per_det):
                self._run_iteration(root)

            # Collect visit counts from root children
            for action, child in root.children.items():
                idx = action_to_idx.get(action)
                if idx is not None:
                    total_visits[idx] += child.visits
                    total_values[idx] += child.total_value

        # Store for dataset
        self.last_mcts_visits = {}
        self.last_mcts_values = {}
        best_idx = 0
        best_v = -1
        for idx in range(len(filtered)):
            v = total_visits[idx]
            if v > 0:
                key = str(filtered[idx])
                self.last_mcts_visits[key] = v
                self.last_mcts_values[key] = total_values[idx] / v
            if v > best_v:
                best_v = v
                best_idx = idx

        if best_v <= 0:
            return self._rng.choice(legal_actions)

        return filtered[best_idx]

    def _filter_root_actions(
        self, state: GameState, actions: list[Action]
    ) -> list[Action]:
        """Pre-filter to top K actions when there are too many."""
        if len(actions) <= self._max_root_actions:
            return list(actions)

        from ..core.actions import ActionType

        scored: list[tuple[float, int]] = []
        for i, action in enumerate(actions):
            score = 0.0
            if action.action_type == ActionType.PLACE and action.position:
                pos = action.position
                center_dist = abs(pos.row - 4.5) + abs(pos.col - 4.5)
                score += max(0, 5 - center_dist)
                # Check if completes sequence (expensive but worth it for filtering)
                new_state = state.apply_action(action)
                if new_state.board.count_sequences(state.current_team) > state.board.count_sequences(state.current_team):
                    score += 10000
            elif action.action_type == ActionType.REMOVE:
                score += 50
            scored.append((score, i))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [actions[i] for _, i in scored[: self._max_root_actions]]

    def _determinize(self, state: GameState) -> GameState:
        """Create a determinized state by sampling plausible opponent hands."""
        from ..core.card import make_full_deck

        team = self._team
        assert team is not None

        new_state = state.copy()

        # Pool = 2 full decks minus our hand
        all_cards = make_full_deck() + make_full_deck()
        pool = list(all_cards)
        for card in state.hands.get(team.value, []):
            if card in pool:
                pool.remove(card)

        self._rng.shuffle(pool)

        pool_idx = 0
        for t in range(state.num_teams):
            if t == team.value:
                continue
            opp_hand_size = len(state.hands.get(t, []))
            if opp_hand_size == 0:
                opp_hand_size = 7 if state.num_teams == 2 else 6
            new_state.hands[t] = pool[pool_idx : pool_idx + opp_hand_size]
            pool_idx += opp_hand_size

        return new_state

    def _run_iteration(self, root: MCTSNode) -> None:
        """Run one MCTS iteration: select -> expand -> simulate -> backpropagate."""
        node = self._select(root)
        node = self._expand(node)
        value = self._simulate(node)
        self._backpropagate(node, value)

    def _select(self, node: MCTSNode) -> MCTSNode:
        c = self._exploration_constant
        while node.is_fully_expanded() and node.children:
            node = node.best_child(c)
        return node

    def _expand(self, node: MCTSNode) -> MCTSNode:
        if not node.untried_actions:
            return node
        action = node.untried_actions.pop()
        new_state = node.state.apply_action(action)
        child = MCTSNode(new_state, parent=node, action=action)
        node.children[action] = child
        return child

    def _simulate(self, node: MCTSNode) -> float:
        """Optimized random rollout from node."""
        state = node.state
        board = state.board.copy()
        hands = {t: list(h) for t, h in state.hands.items()}
        deck = state.deck.copy()
        current_team_val = state.current_team.value
        num_teams = state.num_teams
        seq_to_win = state.sequences_to_win
        rng = self._rng
        team_val = self._team.value
        chips = board.chips  # Direct numpy array reference

        for _ in range(self._rollout_depth):
            # Quick terminal check (direct attribute access, no method call)
            for tv in range(num_teams):
                if len(board._sequences[tv]) >= seq_to_win:
                    return 1.0 if tv == team_val else 0.0

            hand = hands[current_team_val]
            if not hand:
                break

            # Fast random action selection
            rng.shuffle(hand)
            played = False
            team_id = _TEAM_IDS[current_team_val]

            for card in hand:
                if card.is_two_eyed_jack:
                    ep = board.empty_positions
                    if ep:
                        pos = rng.choice(list(ep))
                        hand.remove(card)
                        board.place_chip(pos, team_id)
                        board.check_new_sequences(pos, team_id)
                        drawn = deck.draw()
                        if drawn:
                            hand.append(drawn)
                        played = True
                        break
                elif card.is_one_eyed_jack:
                    continue  # Skip removal in fast rollout
                else:
                    positions = CARD_TO_POSITIONS.get(card)
                    if positions:
                        valid = [p for p in positions if chips[p.row, p.col] == -1]
                        if valid:
                            pos = rng.choice(valid)
                            hand.remove(card)
                            board.place_chip(pos, team_id)
                            board.check_new_sequences(pos, team_id)
                            drawn = deck.draw()
                            if drawn:
                                hand.append(drawn)
                            played = True
                            break

            if not played:
                card = hand.pop()
                deck.discard(card)
                drawn = deck.draw()
                if drawn:
                    hand.append(drawn)

            current_team_val = (current_team_val + 1) % num_teams

        # Truncated — use fast board evaluation
        return _fast_board_eval(chips, team_val, num_teams)

    def _backpropagate(self, node: MCTSNode, value: float) -> None:
        team = self._team
        while node is not None:
            node.visits += 1
            if node.parent is not None and node.parent.state.current_team != team:
                node.total_value += 1.0 - value
            else:
                node.total_value += value
            node = node.parent
