"""CLI entry point. v1 ships server-only: ``open-referee serve``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import uvicorn

from open_referee.config import DEFAULT_CONFIG_PATH, example_config_text, load_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="open-referee", description="Open Referee — AI peer-review harness"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="Start the Open Referee PWA server")
    serve.add_argument("--config", type=Path, default=None, help="Path to config.yaml")
    serve.add_argument("--host", type=str, default=None)
    serve.add_argument("--port", type=int, default=None)

    sub.add_parser("init-config", help="Write an example config and exit")

    args = parser.parse_args(argv)

    if args.command == "init-config":
        if DEFAULT_CONFIG_PATH.exists():
            print(f"Config already exists: {DEFAULT_CONFIG_PATH}", file=sys.stderr)
            return 1
        DEFAULT_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        DEFAULT_CONFIG_PATH.write_text(example_config_text())
        print(f"Wrote {DEFAULT_CONFIG_PATH}")
        return 0

    cfg = load_config(args.config if args.command == "serve" and args.config else None)
    host = args.host or cfg.server.host
    port = args.port or cfg.server.port
    uvicorn.run("open_referee.server.app:create_app", factory=True, host=host, port=port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
