"""Action types for the Sequence game."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .card import Card
from .types import Position


class ActionType(Enum):
    PLACE = "place"
    REMOVE = "remove"
    DEAD_CARD_DISCARD = "dead_card_discard"


@dataclass(frozen=True, slots=True)
class Action:
    card: Card
    position: Position | None  # None for dead card discard
    action_type: ActionType

    def __str__(self) -> str:
        if self.action_type == ActionType.DEAD_CARD_DISCARD:
            return f"Discard({self.card})"
        verb = "Place" if self.action_type == ActionType.PLACE else "Remove"
        return f"{verb}({self.card}@{self.position})"

    def __repr__(self) -> str:
        return str(self)
