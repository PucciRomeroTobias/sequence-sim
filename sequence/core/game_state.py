"""Game state management for Sequence."""

from __future__ import annotations

from .actions import Action, ActionType
from .board import CARD_TO_POSITIONS, Board
from .card import Card
from .deck import Deck
from .types import CORNERS, EMPTY, Position, TeamId


class GameState:
    """Immutable-style game state for Sequence.

    Tracks the board, hands, deck, current team, sequences, and turn number.
    """

    __slots__ = (
        "board",
        "hands",
        "deck",
        "current_team",
        "num_teams",
        "sequences_to_win",
        "turn_number",
    )

    def __init__(
        self,
        board: Board,
        hands: dict[int, list[Card]],
        deck: Deck,
        current_team: TeamId,
        num_teams: int = 2,
        sequences_to_win: int = 2,
        turn_number: int = 0,
    ) -> None:
        self.board = board
        self.hands = hands  # {team_value: [Card, ...]}
        self.deck = deck
        self.current_team = current_team
        self.num_teams = num_teams
        self.sequences_to_win = sequences_to_win
        self.turn_number = turn_number

    def get_legal_actions(self, team: TeamId | None = None) -> list[Action]:
        """Generate all legal actions for the given team (default: current team)."""
        if team is None:
            team = self.current_team
        hand = self.hands[team.value]
        actions: list[Action] = []
        seen: set[tuple[Card, Position | None, ActionType]] = set()

        for card in hand:
            if card.is_two_eyed_jack:
                # Wild: place on any empty position
                for pos in self.board.empty_positions:
                    key = (card, pos, ActionType.PLACE)
                    if key not in seen:
                        seen.add(key)
                        actions.append(Action(card, pos, ActionType.PLACE))
            elif card.is_one_eyed_jack:
                # Remove any opponent chip not part of a completed sequence
                for r in range(10):
                    for c in range(10):
                        pos = Position(r, c)
                        chip = self.board.get_chip(pos)
                        if chip != EMPTY and chip != team.value and chip != 3:
                            # Check this chip isn't part of a completed sequence
                            chip_team = TeamId(chip)
                            if not self.board.is_part_of_own_sequence(pos, chip_team):
                                key = (card, pos, ActionType.REMOVE)
                                if key not in seen:
                                    seen.add(key)
                                    actions.append(
                                        Action(card, pos, ActionType.REMOVE)
                                    )
            else:
                # Normal card: place on one of its board positions
                positions = CARD_TO_POSITIONS.get(card, [])
                is_dead = True
                for pos in positions:
                    if self.board.is_empty(pos):
                        is_dead = False
                        key = (card, pos, ActionType.PLACE)
                        if key not in seen:
                            seen.add(key)
                            actions.append(Action(card, pos, ActionType.PLACE))
                if is_dead:
                    key = (card, None, ActionType.DEAD_CARD_DISCARD)
                    if key not in seen:
                        seen.add(key)
                        actions.append(
                            Action(card, None, ActionType.DEAD_CARD_DISCARD)
                        )

        return actions

    def apply_action(self, action: Action) -> GameState:
        """Apply an action and return a new GameState."""
        new_board = self.board.copy()
        new_hands = {t: list(h) for t, h in self.hands.items()}
        new_deck = self.deck.copy()
        team = self.current_team
        new_sequences_formed = 0

        # Remove the played card from hand
        new_hands[team.value].remove(action.card)

        if action.action_type == ActionType.PLACE:
            assert action.position is not None
            new_board.place_chip(action.position, team)
            new_seqs = new_board.check_new_sequences(action.position, team)
            new_sequences_formed = len(new_seqs)
        elif action.action_type == ActionType.REMOVE:
            assert action.position is not None
            new_board.remove_chip(action.position)
        elif action.action_type == ActionType.DEAD_CARD_DISCARD:
            new_deck.discard(action.card)

        # Draw a new card
        drawn = new_deck.draw()
        if drawn is not None:
            new_hands[team.value].append(drawn)

        # Next team
        next_team_val = (team.value + 1) % self.num_teams
        next_team = TeamId(next_team_val)

        new_state = GameState(
            board=new_board,
            hands=new_hands,
            deck=new_deck,
            current_team=next_team,
            num_teams=self.num_teams,
            sequences_to_win=self.sequences_to_win,
            turn_number=self.turn_number + 1,
        )
        return new_state

    def is_terminal(self) -> TeamId | None:
        """Check if any team has won. Returns winning TeamId or None."""
        for team in TeamId:
            if team.value >= self.num_teams:
                continue
            if self.board.count_sequences(team) >= self.sequences_to_win:
                return team
        return None

    def get_visible_state(self, team: TeamId) -> GameState:
        """Return a copy where opponent hands are hidden (empty lists)."""
        visible_hands: dict[int, list[Card]] = {}
        for t, hand in self.hands.items():
            if t == team.value:
                visible_hands[t] = list(hand)
            else:
                visible_hands[t] = []
        return GameState(
            board=self.board.copy(),
            hands=visible_hands,
            deck=self.deck.copy(),
            current_team=self.current_team,
            num_teams=self.num_teams,
            sequences_to_win=self.sequences_to_win,
            turn_number=self.turn_number,
        )

    def copy(self) -> GameState:
        return GameState(
            board=self.board.copy(),
            hands={t: list(h) for t, h in self.hands.items()},
            deck=self.deck.copy(),
            current_team=self.current_team,
            num_teams=self.num_teams,
            sequences_to_win=self.sequences_to_win,
            turn_number=self.turn_number,
        )
