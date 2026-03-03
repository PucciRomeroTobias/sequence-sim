"""Card tracking for perfect public information in Sequence games."""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

from .board import CARD_TO_POSITIONS, LAYOUT
from .card import Card, make_full_deck
from .types import CORNERS, EMPTY, CORNER, Position, TeamId

if TYPE_CHECKING:
    from .actions import Action


# Total copies of each unique card in a 2-deck Sequence game
_COPIES_PER_CARD = 2

# All 96 non-corner board positions (10x10 minus 4 corners)
_BOARD_POSITIONS: frozenset[Position] = frozenset(
    Position(r, c)
    for r in range(10)
    for c in range(10)
    if Position(r, c) not in CORNERS
)


class CardTracker:
    """Tracks publicly visible card information throughout a Sequence game.

    Maintains counts of played/discarded cards to answer questions about:
    - Which cards are still available in the deck/opponents' hands
    - Which board positions are permanently dead (unreachable)
    - Which positions are guaranteed (you hold all remaining copies)
    - Probability estimates for opponent holdings
    """

    __slots__ = (
        "_team",
        "_num_teams",
        "_played",
        "_discarded",
        "_hand",
        "_full_deck_counts",
    )

    def __init__(self, team: TeamId, num_teams: int = 2) -> None:
        self._team = team
        self._num_teams = num_teams
        # Count of each card that has been played onto the board
        self._played: Counter[Card] = Counter()
        # Count of each card that has been discarded (dead card discards)
        self._discarded: Counter[Card] = Counter()
        # Our current hand (synced each turn)
        self._hand: list[Card] = []
        # Pre-compute: how many copies of each unique card exist in 2 decks
        self._full_deck_counts: Counter[Card] = Counter()
        for card in make_full_deck():
            self._full_deck_counts[card] += _COPIES_PER_CARD

    def on_action(self, action: Action, team: TeamId) -> None:
        """Update tracking based on a played action (called for all players)."""
        from .actions import ActionType

        card = action.card
        if action.action_type == ActionType.PLACE:
            self._played[card] += 1
        elif action.action_type == ActionType.DEAD_CARD_DISCARD:
            self._discarded[card] += 1
        elif action.action_type == ActionType.REMOVE:
            # One-eyed jack used to remove — the jack card itself is consumed
            self._played[card] += 1

    def sync_hand(self, hand: list[Card]) -> None:
        """Sync our known hand (called at the start of each choose_action)."""
        self._hand = list(hand)

    def copies_played(self, card: Card) -> int:
        """How many copies of this card have been played onto the board."""
        return self._played[card]

    def copies_discarded(self, card: Card) -> int:
        """How many copies of this card have been discarded as dead cards."""
        return self._discarded[card]

    def copies_used(self, card: Card) -> int:
        """Total copies of this card that are no longer in the card pool."""
        return self._played[card] + self._discarded[card]

    def copies_in_hand(self, card: Card) -> int:
        """How many copies of this card are in our hand."""
        return sum(1 for c in self._hand if c == card)

    def copies_remaining_in_pool(self, card: Card) -> int:
        """How many copies of this card could still be in deck or opponent hands.

        pool = total_copies - used - in_our_hand
        """
        total = self._full_deck_counts.get(card, 0)
        used = self.copies_used(card)
        in_hand = self.copies_in_hand(card)
        return max(0, total - used - in_hand)

    def is_position_permanently_dead(self, pos: Position, board_chips) -> bool:
        """Check if a position can never be filled by any remaining card.

        A position is dead if:
        - It's currently empty, AND
        - The card for that position has all copies used (played/discarded), AND
        - No two-eyed jacks remain in the pool (they could fill any position)

        Note: Corners are never dead (they're pre-filled as wild).
        """
        if pos in CORNERS:
            return False
        if int(board_chips[pos.row, pos.col]) != EMPTY:
            return False  # Already occupied — not "dead"

        # Find which card maps to this position
        layout_card = LAYOUT[pos.row][pos.col]
        if layout_card is None:
            return False  # Corner

        # Check if all copies of that card are gone
        total = self._full_deck_counts.get(layout_card, 0)
        used = self.copies_used(layout_card)
        in_hand = self.copies_in_hand(layout_card)
        remaining_of_card = total - used - in_hand

        if remaining_of_card > 0:
            return False

        # Also check if we hold any copies (we could still play it)
        if in_hand > 0:
            return False

        # Check if any two-eyed jacks remain anywhere
        from .types import Rank, Suit

        for suit in (Suit.DIAMONDS, Suit.CLUBS):
            tej = Card(Rank.JACK, suit)
            if self.copies_remaining_in_pool(tej) > 0:
                return False
            if self.copies_in_hand(tej) > 0:
                return False

        return True

    def get_guaranteed_positions(self, hand: list[Card], board_chips) -> set[Position]:
        """Positions where we hold ALL remaining copies of the needed card.

        If we hold both copies (and the position is empty), we're guaranteed
        to be able to place there eventually.
        """
        guaranteed: set[Position] = set()
        hand_counts: Counter[Card] = Counter(hand)

        for card, count in hand_counts.items():
            if card.is_jack:
                continue
            positions = CARD_TO_POSITIONS.get(card, [])
            for pos in positions:
                if int(board_chips[pos.row, pos.col]) != EMPTY:
                    continue
                # We hold `count` copies. How many total remain outside our hand?
                total = self._full_deck_counts.get(card, 0)
                used = self.copies_used(card)
                total_remaining = total - used  # includes our hand
                if count >= total_remaining:
                    guaranteed.add(pos)

        return guaranteed

    def opponent_has_card_probability(self, card: Card) -> float:
        """Rough probability that any opponent holds at least one copy of this card.

        Based on the ratio of remaining copies to total unknown cards.
        """
        remaining = self.copies_remaining_in_pool(card)
        if remaining == 0:
            return 0.0
        total_unknown = self._total_unknown_pool_size()
        if total_unknown == 0:
            return 0.0
        # Approximate: P(at least one in opponent hands)
        # Opponent hand slots = (num_teams - 1) * hand_size (typically 7)
        opp_hand_slots = (self._num_teams - 1) * 7
        if opp_hand_slots >= total_unknown:
            return 1.0 if remaining > 0 else 0.0
        # 1 - P(none in opp hands) ≈ 1 - C(pool-remaining, slots)/C(pool, slots)
        # Simplified: 1 - product((pool-remaining-i)/(pool-i) for i in range(slots))
        p_none = 1.0
        pool = total_unknown
        non_target = pool - remaining
        for i in range(min(opp_hand_slots, pool)):
            if pool - i <= 0:
                break
            p_none *= max(0, non_target - i) / (pool - i)
        return 1.0 - p_none

    def get_unknown_card_pool(self) -> list[Card]:
        """Cards that could be in the deck or opponent hands.

        = 2 full decks - played - discarded - our hand
        Used for MCTS determinization.
        """
        pool: list[Card] = []
        hand_counts: Counter[Card] = Counter(self._hand)

        for card, total in self._full_deck_counts.items():
            remaining = total - self.copies_used(card)
            # Subtract our hand copies
            remaining -= hand_counts.get(card, 0)
            for _ in range(max(0, remaining)):
                pool.append(card)

        return pool

    def _total_unknown_pool_size(self) -> int:
        """Total number of cards in the unknown pool (deck + opponent hands)."""
        total = 0
        hand_counts: Counter[Card] = Counter(self._hand)
        for card, deck_total in self._full_deck_counts.items():
            remaining = deck_total - self.copies_used(card) - hand_counts.get(card, 0)
            total += max(0, remaining)
        return total
