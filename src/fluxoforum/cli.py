"""Command-line entry point."""

from __future__ import annotations

import argparse
import json
import os

from .diagnostics import system_diagnostics
from .ui import create_app


def main() -> None:
    parser = argparse.ArgumentParser(prog="fluxoforum")
    parser.add_argument("--host", default=os.getenv("FLUXOFORUM_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("FLUXOFORUM_PORT", "7860")))
    parser.add_argument(
        "--data-root", default=os.getenv("FLUXOFORUM_DATA_ROOT", "/workspace/fluxoforum-data")
    )
    parser.add_argument("--diagnostics", action="store_true")
    args = parser.parse_args()
    if args.diagnostics:
        print(json.dumps(system_diagnostics(args.data_root, check_model_access=True), indent=2))
        return
    auth = None
    username = os.getenv("FLUXOFORUM_USERNAME")
    password = os.getenv("FLUXOFORUM_PASSWORD")
    if username and password:
        auth = (username, password)
    print(json.dumps(system_diagnostics(args.data_root, check_model_access=True), indent=2))
    app = create_app(args.data_root)
    app.queue(default_concurrency_limit=1, max_size=8).launch(
        server_name=args.host,
        server_port=args.port,
        auth=auth,
        show_error=True,
        allowed_paths=[args.data_root],
    )
