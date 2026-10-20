"""Run the Phase 1 APPS lineup: ``python -m agent_control.apps_backdoor.run``."""

from __future__ import annotations

import argparse
import dataclasses
from collections.abc import Sequence
from pathlib import Path

from agent_control.apps_backdoor.experiment import (
    PROTOCOL_MODES,
    build_protocols,
    build_setting,
)
from agent_control.config import ExperimentConfig
from agent_control.runner import build_tasks, require_docker, run_tasks


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--protocols",
        nargs="+",
        choices=sorted(PROTOCOL_MODES),
        default=None,
        help="subset of the lineup (default: all)",
    )
    p.add_argument("--limit", type=int, default=None, help="samples per task")
    p.add_argument("--untrusted-model", default=None)
    p.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="log directory (use a fresh one so analyze only sees this run)",
    )
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    config = ExperimentConfig()
    if args.untrusted_model:
        config = dataclasses.replace(config, untrusted_model=args.untrusted_model)
    if args.log_dir:
        config = dataclasses.replace(config, log_dir=args.log_dir)
    config.require_api_key()
    require_docker()
    protocols = build_protocols(config)
    if args.protocols:
        protocols = {k: v for k, v in protocols.items() if k in args.protocols}
    tasks = build_tasks(build_setting(config), protocols, PROTOCOL_MODES, config)
    logs = run_tasks(tasks, config, limit=args.limit)
    print(f"Ran {len(logs)} tasks; logs in {config.log_dir}")


if __name__ == "__main__":
    main()
