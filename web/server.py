#!/usr/bin/env python3
"""Web server for Sequence game: Human vs SmartAgent."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio
import random as stdlib_random

import socketio
from aiohttp import web

from sequence.agents.smart_agent import SmartAgent

# Try to use NeuralAgent if model is available
_USE_NEURAL = False
try:
    from sequence.agents.neural_agent import NeuralAgent
    _NEURAL_MODEL_PATH = str(Path(__file__).resolve().parent.parent / "data" / "nn" / "model.pt")
    if Path(_NEURAL_MODEL_PATH).exists():
        _USE_NEURAL = True
except ImportError:
    pass
from sequence.core.board import LAYOUT
from sequence.core.card_tracker import CardTracker
from sequence.core.deck import Deck
from sequence.core.game import GameConfig
from sequence.core.game_state import GameState
from sequence.core.board import Board
from sequence.core.types import TeamId, Position, EMPTY, CORNER
from sequence.core.actions import Action, ActionType
from sequence.core.card import Card

# Board layout as strings for the frontend
BOARD_LAYOUT = []
for r in range(10):
    row = []
    for c in range(10):
        card = LAYOUT[r][c]
        row.append(str(card) if card else "FREE")
    BOARD_LAYOUT.append(row)

sio = socketio.AsyncServer(cors_allowed_origins="*", async_mode="aiohttp")
app = web.Application()
sio.attach(app)

# Game sessions: sid -> game state
games: dict[str, dict] = {}


def serialize_state(session: dict) -> dict:
    """Convert game state to JSON for the frontend."""
    state: GameState = session["state"]
    board = state.board
    chips = board.chips

    # Board cells: 10x10 array with chip info
    board_cells = []
    for r in range(10):
        row = []
        for c in range(10):
            val = int(chips[r, c])
            cell = {"card": BOARD_LAYOUT[r][c]}
            if val == CORNER:
                cell["chip"] = "corner"
            elif val == EMPTY:
                cell["chip"] = None
            else:
                cell["chip"] = "blue" if val == 0 else "red"
            # Mark cells in completed sequences (and which team)
            cell["inSequence"] = None
            for team_val in range(2):
                tid = TeamId(team_val)
                for seq in board.get_all_sequences(tid):
                    if Position(r, c) in seq:
                        cell["inSequence"] = "blue" if team_val == 0 else "red"
            row.append(cell)
        board_cells.append(row)

    # Player hand (team 0 = human = blue)
    human_team = session["human_team"]
    hand = state.hands.get(human_team.value, [])
    hand_cards = [str(c) for c in hand]

    # Legal actions for human
    legal = []
    if state.current_team == human_team:
        actions = state.get_legal_actions(human_team)
        for a in actions:
            legal.append({
                "card": str(a.card),
                "position": [a.position.row, a.position.col] if a.position else None,
                "type": a.action_type.value,
            })

    # Scores
    scores = {
        "blue": board.count_sequences(TeamId.TEAM_0),
        "red": board.count_sequences(TeamId.TEAM_1),
    }

    winner = state.is_terminal()

    # Last AI move
    last_ai_move = session.get("last_ai_move")

    return {
        "board": board_cells,
        "hand": hand_cards,
        "legalActions": legal,
        "currentTeam": "blue" if state.current_team == TeamId.TEAM_0 else "red",
        "humanTeam": "blue" if human_team == TeamId.TEAM_0 else "red",
        "scores": scores,
        "winner": ("blue" if winner == TeamId.TEAM_0 else "red") if winner is not None else None,
        "turnNumber": state.turn_number,
        "lastAiMove": last_ai_move,
        "deckCount": state.deck.remaining,
    }


def create_game(sid: str) -> dict:
    """Create a new game session."""
    config = GameConfig(num_teams=2, sequences_to_win=2)
    board = Board()
    deck = Deck()
    hands: dict[int, list] = {}
    for t in range(2):
        hand = []
        for _ in range(7):
            c = deck.draw()
            if c is not None:
                hand.append(c)
        hands[t] = hand

    state = GameState(
        board=board, hands=hands, deck=deck,
        current_team=TeamId.TEAM_0, num_teams=2, sequences_to_win=2,
    )

    human_team = TeamId.TEAM_0
    ai_team = TeamId.TEAM_1

    if _USE_NEURAL:
        agent = NeuralAgent(_NEURAL_MODEL_PATH, use_lookahead=True)
    else:
        agent = SmartAgent(use_lookahead=True)
    agent.notify_game_start(ai_team, config)

    tracker = CardTracker(ai_team, 2)

    session = {
        "state": state,
        "human_team": human_team,
        "ai_team": ai_team,
        "agent": agent,
        "tracker": tracker,
        "config": config,
        "last_ai_move": None,
    }
    games[sid] = session
    return session


@sio.event
async def connect(sid, environ):
    print(f"Client connected: {sid}")


@sio.event
async def disconnect(sid):
    print(f"Client disconnected: {sid}")
    games.pop(sid, None)


@sio.event
async def new_game(sid):
    """Start a new game."""
    session = create_game(sid)
    await sio.emit("gameState", serialize_state(session), to=sid)


@sio.event
async def play_action(sid, data):
    """Human plays an action."""
    session = games.get(sid)
    if not session:
        return

    state: GameState = session["state"]
    human_team = session["human_team"]

    # Verify it's human's turn
    if state.current_team != human_team:
        return

    # Parse the action
    card_str = data["card"]
    position = data.get("position")
    action_type = data["type"]

    card = Card.from_str(card_str)
    pos = Position(position[0], position[1]) if position else None
    at = ActionType(action_type)
    action = Action(card=card, position=pos, action_type=at)

    # Verify it's legal
    legal_actions = state.get_legal_actions(human_team)
    if action not in legal_actions:
        await sio.emit("error", {"message": "Illegal action"}, to=sid)
        return

    # Apply human action
    session["agent"].notify_action(action, human_team)
    session["tracker"].on_action(action, human_team)
    state = state.apply_action(action)
    session["state"] = state

    # Check if game ended
    winner = state.is_terminal()
    if winner is not None:
        await sio.emit("gameState", serialize_state(session), to=sid)
        return

    # Send intermediate state (show human's move, AI is thinking)
    await sio.emit("gameState", serialize_state(session), to=sid)

    # AI's turn with random delay (3-8 seconds)
    ai_team = session["ai_team"]
    if state.current_team == ai_team:
        ai_legal = state.get_legal_actions(ai_team)
        visible = state.get_visible_state(ai_team)
        ai_action = session["agent"].choose_action(visible, ai_legal)

        # Random delay to feel more natural
        delay = stdlib_random.uniform(3.0, 8.0)
        await asyncio.sleep(delay)

        # Record AI move for display
        session["last_ai_move"] = {
            "card": str(ai_action.card),
            "position": [ai_action.position.row, ai_action.position.col] if ai_action.position else None,
            "type": ai_action.action_type.value,
        }

        session["agent"].notify_action(ai_action, ai_team)
        session["tracker"].on_action(ai_action, ai_team)
        state = state.apply_action(ai_action)
        session["state"] = state

    await sio.emit("gameState", serialize_state(session), to=sid)


@sio.event
async def get_board_layout(sid):
    """Send the static board layout."""
    await sio.emit("boardLayout", BOARD_LAYOUT, to=sid)


# Serve static files (card images)
app.router.add_static("/cards/", path=Path(__file__).parent / "public" / "cards")

if __name__ == "__main__":
    print("Starting Sequence web server on http://localhost:8080")
    print("Frontend should connect from http://localhost:5173")
    web.run_app(app, host="0.0.0.0", port=8080)
