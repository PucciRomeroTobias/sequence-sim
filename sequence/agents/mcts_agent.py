"""MCTS agent with Information Set determinization for imperfect information."""

from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING

from ..core.types import TeamId
from .base import Agent

if TYPE_CHECKING:
    from ..core.actions import Action
    from ..core.game import GameConfig
    from ..core.game_state import GameState


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
        exploitation = self.total_value / self.visits
        exploration = exploration_constant * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )
        return exploitation + exploration

    def best_child(self, c: float) -> MCTSNode:
        return max(self.children.values(), key=lambda n: n.ucb1(c))

    def is_fully_expanded(self) -> bool:
        return len(self.untried_actions) == 0


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

        # Aggregate visit counts across determinizations
        total_visits: dict[int, int] = {}  # action index -> total visits
        total_values: dict[int, float] = {}

        iters_per_det = max(1, self._iterations // self._num_determinizations)

        for _ in range(self._num_determinizations):
            det_state = self._determinize(state)
            root = MCTSNode(det_state, legal_actions=list(filtered))

            for _ in range(iters_per_det):
                self._run_iteration(root)

            # Collect visit counts from root children
            for action, child in root.children.items():
                try:
                    idx = filtered.index(action)
                except ValueError:
                    continue
                total_visits[idx] = total_visits.get(idx, 0) + child.visits
                total_values[idx] = total_values.get(idx, 0.0) + child.total_value

        # Store for dataset
        self.last_mcts_visits = {}
        self.last_mcts_values = {}
        for idx, visits in total_visits.items():
            key = str(filtered[idx])
            self.last_mcts_visits[key] = visits
            self.last_mcts_values[key] = (
                total_values.get(idx, 0.0) / visits if visits > 0 else 0.0
            )

        # Choose action with most visits
        if not total_visits:
            return self._rng.choice(legal_actions)

        best_idx = max(total_visits, key=total_visits.get)
        return filtered[best_idx]

    def _filter_root_actions(
        self, state: GameState, actions: list[Action]
    ) -> list[Action]:
        """Pre-filter to top K actions when there are too many."""
        if len(actions) <= self._max_root_actions:
            return list(actions)

        from ..core.actions import ActionType

        # Quick scoring for filtering
        scored: list[tuple[float, int]] = []
        for i, action in enumerate(actions):
            score = 0.0
            if action.action_type == ActionType.PLACE and action.position:
                pos = action.position
                center_dist = abs(pos.row - 4.5) + abs(pos.col - 4.5)
                score += max(0, 5 - center_dist)
                # Check if completes sequence
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
        from ..core.types import TeamId

        team = self._team
        assert team is not None

        new_state = state.copy()

        # Determine which cards are "seen" (in our hand, on board as chips don't
        # correspond to cards, or discarded)
        # For simplicity: cards in our hand are known. The unseen cards pool is
        # all cards minus our hand minus cards we've tracked.
        # Since we don't track discards perfectly, we use a conservative approach:
        # pool = 2 full decks - our hand
        all_cards = make_full_deck() + make_full_deck()

        # Remove our hand from pool
        pool = list(all_cards)
        for card in state.hands.get(team.value, []):
            if card in pool:
                pool.remove(card)

        # Remove cards known to be in the deck's draw pile + discard
        # (We don't have perfect info, but we approximate)
        self._rng.shuffle(pool)

        # Assign hands to opponents from pool
        pool_idx = 0
        for t in range(state.num_teams):
            if t == team.value:
                continue
            opp_hand_size = len(state.hands.get(t, []))
            # If we don't know the hand size, use a default
            if opp_hand_size == 0:
                opp_hand_size = 7 if state.num_teams == 2 else 6
            hand = pool[pool_idx : pool_idx + opp_hand_size]
            pool_idx += opp_hand_size
            new_state.hands[t] = hand

        return new_state

    def _run_iteration(self, root: MCTSNode) -> None:
        """Run one MCTS iteration: select -> expand -> simulate -> backpropagate."""
        node = self._select(root)
        node = self._expand(node)
        value = self._simulate(node)
        self._backpropagate(node, value)

    def _select(self, node: MCTSNode) -> MCTSNode:
        """Select a promising node using UCB1."""
        while node.is_fully_expanded() and node.children:
            node = node.best_child(self._exploration_constant)
        return node

    def _expand(self, node: MCTSNode) -> MCTSNode:
        """Expand by adding one untried action."""
        if not node.untried_actions:
            return node

        action = node.untried_actions.pop()
        new_state = node.state.apply_action(action)
        child = MCTSNode(new_state, parent=node, action=action)
        node.children[action] = child
        return child

    def _simulate(self, node: MCTSNode) -> float:
        """Random rollout from node, return value for our team."""
        from ..core.actions import Action, ActionType
        from ..core.board import CARD_TO_POSITIONS
        from ..core.types import EMPTY

        state = node.state

        # Light rollout: instead of full copy + apply_action loop,
        # work on a mutable copy and use fast random action selection
        board = state.board.copy()
        hands = {t: list(h) for t, h in state.hands.items()}
        deck = state.deck.copy()
        current_team_val = state.current_team.value
        num_teams = state.num_teams
        rng = self._rng

        for _ in range(self._rollout_depth):
            # Quick terminal check
            for tv in range(num_teams):
                if board.count_sequences(TeamId(tv)) >= state.sequences_to_win:
                    return 1.0 if tv == self._team.value else 0.0

            hand = hands[current_team_val]
            if not hand:
                break

            # Fast random action: pick random card from hand, try to play it
            rng.shuffle(hand)
            played = False
            for card in hand:
                if card.is_two_eyed_jack:
                    empty = list(board.empty_positions)
                    if empty:
                        pos = rng.choice(empty)
                        hand.remove(card)
                        board.place_chip(pos, TeamId(current_team_val))
                        board.check_new_sequences(pos, TeamId(current_team_val))
                        drawn = deck.draw()
                        if drawn:
                            hand.append(drawn)
                        played = True
                        break
                elif card.is_one_eyed_jack:
                    # Skip one-eyed jacks in fast rollout for speed
                    continue
                else:
                    positions = CARD_TO_POSITIONS.get(card, [])
                    valid = [p for p in positions if board.is_empty(p)]
                    if valid:
                        pos = rng.choice(valid)
                        hand.remove(card)
                        board.place_chip(pos, TeamId(current_team_val))
                        board.check_new_sequences(pos, TeamId(current_team_val))
                        drawn = deck.draw()
                        if drawn:
                            hand.append(drawn)
                        played = True
                        break

            if not played:
                # Discard a random card as dead
                card = hand.pop()
                deck.discard(card)
                drawn = deck.draw()
                if drawn:
                    hand.append(drawn)

            current_team_val = (current_team_val + 1) % num_teams

        # Truncated — use heuristic evaluation
        team = self._team
        assert team is not None
        own_seqs = board.count_sequences(team)
        opp_seqs = max(
            (board.count_sequences(TeamId(t)) for t in range(num_teams) if t != team.value),
            default=0,
        )
        diff = own_seqs - opp_seqs
        return 0.5 + 0.25 * diff

    def _backpropagate(self, node: MCTSNode, value: float) -> None:
        """Propagate the simulation result up the tree."""
        while node is not None:
            node.visits += 1
            # Value is from our team's perspective
            # For opponent nodes, invert the value
            if (
                node.parent is not None
                and node.parent.state.current_team != self._team
            ):
                node.total_value += 1.0 - value
            else:
                node.total_value += value
            node = node.parent
