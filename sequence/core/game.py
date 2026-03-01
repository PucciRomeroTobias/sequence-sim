"""Game orchestrator for Sequence."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from .board import Board
from .card import Card
from .deck import Deck
from .game_state import GameState
from .types import TeamId

if TYPE_CHECKING:
    from ..agents.base import Agent


@dataclass
class GameConfig:
    num_teams: int = 2
    hand_size: int = 7  # 7 for 2 teams, 6 for 3
    sequences_to_win: int = 2  # 2 for 2 teams, 1 for 3
    seed: int | None = None
    max_turns: int = 500  # Safety limit

    def __post_init__(self) -> None:
        if self.num_teams == 3:
            self.hand_size = 6
            self.sequences_to_win = 1


@dataclass
class MoveRecord:
    turn: int
    team: int
    action: dict[str, Any]  # Serializable action
    legal_actions_count: int
    hand_before: list[str]  # Card strings
    board_snapshot: list[list[int]]
    card_drawn: str | None
    sequences_before: dict[int, int]
    sequences_after: dict[int, int]
    thinking_time_ms: float = 0.0
    mcts_visits: dict[str, int] | None = None  # Optional MCTS data
    mcts_values: dict[str, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn": self.turn,
            "team": self.team,
            "action": self.action,
            "legal_actions_count": self.legal_actions_count,
            "hand_before": self.hand_before,
            "board_snapshot": self.board_snapshot,
            "card_drawn": self.card_drawn,
            "sequences_before": self.sequences_before,
            "sequences_after": self.sequences_after,
            "thinking_time_ms": self.thinking_time_ms,
            "mcts_visits": self.mcts_visits,
            "mcts_values": self.mcts_values,
        }


@dataclass
class GameRecord:
    game_id: str
    seed: int | None
    agent_names: list[str]
    config: dict[str, Any]
    winner: int | None
    total_turns: int
    moves: list[MoveRecord] = field(default_factory=list)
    duration_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "game_id": self.game_id,
            "seed": self.seed,
            "agent_names": self.agent_names,
            "config": self.config,
            "winner": self.winner,
            "total_turns": self.total_turns,
            "moves": [m.to_dict() for m in self.moves],
            "duration_ms": self.duration_ms,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GameRecord:
        moves = [MoveRecord(**m) for m in d["moves"]]
        return cls(
            game_id=d["game_id"],
            seed=d["seed"],
            agent_names=d["agent_names"],
            config=d["config"],
            winner=d["winner"],
            total_turns=d["total_turns"],
            moves=moves,
            duration_ms=d.get("duration_ms", 0.0),
        )


def _serialize_action(action: Any) -> dict[str, Any]:
    return {
        "card": str(action.card),
        "position": list(action.position) if action.position else None,
        "action_type": action.action_type.value,
    }


class Game:
    """Orchestrates a Sequence game between agents."""

    def __init__(
        self,
        agent_factories: list[Callable[[], Agent]],
        config: GameConfig | None = None,
    ) -> None:
        self.config = config or GameConfig()
        assert len(agent_factories) == self.config.num_teams
        self._agent_factories = agent_factories

    def play(self) -> GameRecord:
        start_time = time.monotonic()
        config = self.config
        game_id = str(uuid.uuid4())[:8]

        # Create agents
        agents: list[Agent] = [f() for f in self._agent_factories]
        agent_names = [type(a).__name__ for a in agents]

        # Initialize game state
        board = Board()
        deck = Deck(seed=config.seed)
        hands: dict[int, list[Card]] = {}
        for t in range(config.num_teams):
            hand: list[Card] = []
            for _ in range(config.hand_size):
                card = deck.draw()
                if card is not None:
                    hand.append(card)
            hands[t] = hand

        state = GameState(
            board=board,
            hands=hands,
            deck=deck,
            current_team=TeamId(0),
            num_teams=config.num_teams,
            sequences_to_win=config.sequences_to_win,
        )

        # Notify agents of game start
        for i, agent in enumerate(agents):
            agent.notify_game_start(TeamId(i), config)

        moves: list[MoveRecord] = []
        winner: TeamId | None = None

        for turn in range(config.max_turns):
            team = state.current_team
            agent = agents[team.value]

            # Get legal actions
            legal_actions = state.get_legal_actions(team)
            if not legal_actions:
                # No legal actions — skip turn (shouldn't normally happen)
                # Advance to next team
                next_val = (team.value + 1) % config.num_teams
                state.current_team = TeamId(next_val)
                state.turn_number += 1
                continue

            # Record state before action
            hand_before = [str(c) for c in state.hands[team.value]]
            board_snapshot = state.board.to_list()
            sequences_before = {
                t: state.board.count_sequences(TeamId(t))
                for t in range(config.num_teams)
            }

            # Agent chooses action
            t0 = time.monotonic()
            visible_state = state.get_visible_state(team)
            action = agent.choose_action(visible_state, legal_actions)
            thinking_ms = (time.monotonic() - t0) * 1000

            # Get MCTS data if available
            mcts_visits = getattr(agent, "last_mcts_visits", None)
            mcts_values = getattr(agent, "last_mcts_values", None)

            # Record the hand before to find drawn card
            old_hand_set = list(state.hands[team.value])

            # Apply action
            state = state.apply_action(action)

            # Figure out which card was drawn
            new_hand = state.hands[team.value]
            card_drawn = None
            # Compare hand sizes: if same size, a card was drawn
            old_after_play = [c for c in old_hand_set if c != action.card]
            # Actually find the new card
            old_counts: dict[Card, int] = {}
            for c in old_after_play:
                old_counts[c] = old_counts.get(c, 0) + 1
            new_counts: dict[Card, int] = {}
            for c in new_hand:
                new_counts[c] = new_counts.get(c, 0) + 1
            for c, cnt in new_counts.items():
                if cnt > old_counts.get(c, 0):
                    card_drawn = str(c)
                    break

            sequences_after = {
                t: state.board.count_sequences(TeamId(t))
                for t in range(config.num_teams)
            }

            move = MoveRecord(
                turn=turn,
                team=team.value,
                action=_serialize_action(action),
                legal_actions_count=len(legal_actions),
                hand_before=hand_before,
                board_snapshot=board_snapshot,
                card_drawn=card_drawn,
                sequences_before=sequences_before,
                sequences_after=sequences_after,
                thinking_time_ms=thinking_ms,
                mcts_visits=mcts_visits,
                mcts_values=mcts_values,
            )
            moves.append(move)

            # Notify all agents
            for a in agents:
                a.notify_action(action, team)

            # Check for winner
            winner = state.is_terminal()
            if winner is not None:
                break

        elapsed_ms = (time.monotonic() - start_time) * 1000

        return GameRecord(
            game_id=game_id,
            seed=config.seed,
            agent_names=agent_names,
            config={
                "num_teams": config.num_teams,
                "hand_size": config.hand_size,
                "sequences_to_win": config.sequences_to_win,
                "max_turns": config.max_turns,
            },
            winner=winner.value if winner is not None else None,
            total_turns=len(moves),
            moves=moves,
            duration_ms=elapsed_ms,
        )
