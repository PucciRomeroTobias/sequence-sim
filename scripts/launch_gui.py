#!/usr/bin/env python3
"""Launch the Sequence game GUI."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    parser = argparse.ArgumentParser(description="Launch Sequence GUI")
    parser.add_argument(
        "--replay",
        type=str,
        default=None,
        help="Path to a JSONL game record file to replay",
    )
    parser.add_argument(
        "--live",
        type=str,
        default=None,
        help="Comma-separated agent names for live mode (e.g., 'random,greedy')",
    )
    args = parser.parse_args()

    from sequence.gui.app import SequenceApp

    app = SequenceApp()

    if args.replay:
        app.load_replay(args.replay)
    elif args.live:
        agents = [n.strip() for n in args.live.split(",")]
        if len(agents) == 2:
            app.start_live(agents[0], agents[1])

    app.mainloop()


if __name__ == "__main__":
    main()
