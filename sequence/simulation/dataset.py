"""Dataset I/O for game records (JSONL format)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from ..core.game import GameRecord


class DatasetWriter:
    """Write game records to a JSONL file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._file = None

    def __enter__(self) -> DatasetWriter:
        self._file = open(self.path, "a", encoding="utf-8")
        return self

    def __exit__(self, *args: object) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def write(self, record: GameRecord) -> None:
        """Write a single game record as a JSONL line."""
        if self._file is None:
            raise RuntimeError("DatasetWriter must be used as a context manager")
        line = json.dumps(record.to_dict(), separators=(",", ":"))
        self._file.write(line + "\n")

    def write_many(self, records: list[GameRecord]) -> None:
        """Write multiple game records."""
        for record in records:
            self.write(record)


class DatasetReader:
    """Read game records from a JSONL file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def __iter__(self) -> Iterator[GameRecord]:
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    d = json.loads(line)
                    yield GameRecord.from_dict(d)

    def read_all(self) -> list[GameRecord]:
        """Read all game records into a list."""
        return list(self)


def to_move_dataframe(records: list[GameRecord]):
    """Convert game records to a pandas DataFrame with one row per move.

    Columns: game_id, turn, team, action_type, card, position_row, position_col,
    legal_actions_count, thinking_time_ms, winner, sequences_before_*, sequences_after_*

    Returns:
        A pandas DataFrame.
    """
    import pandas as pd

    rows = []
    for rec in records:
        for move in rec.moves:
            row = {
                "game_id": rec.game_id,
                "turn": move.turn,
                "team": move.team,
                "action_type": move.action["action_type"],
                "card": move.action["card"],
                "position_row": move.action["position"][0] if move.action["position"] else None,
                "position_col": move.action["position"][1] if move.action["position"] else None,
                "legal_actions_count": move.legal_actions_count,
                "thinking_time_ms": move.thinking_time_ms,
                "winner": rec.winner,
            }
            for t, count in move.sequences_before.items():
                row[f"sequences_before_{t}"] = count
            for t, count in move.sequences_after.items():
                row[f"sequences_after_{t}"] = count
            rows.append(row)
    return pd.DataFrame(rows)
