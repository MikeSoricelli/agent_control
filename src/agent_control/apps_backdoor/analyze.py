"""Summarise logs and save plots: ``python -m agent_control.apps_backdoor.analyze``."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from agent_control.analysis import (
    load_results,
    plot_summary,
    plot_suspicion,
    summarize,
)
from agent_control.config import ExperimentConfig


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--log-dir", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=Path("results"))
    args = p.parse_args(argv)
    config = ExperimentConfig()
    log_dir = args.log_dir or config.log_dir

    df = load_results(log_dir)
    summary = summarize(
        df,
        config.audit_budget,
        trusted_only="trusted_only",
        untrusted_only="untrusted_only",
        upfront_deployment_samples=config.upfront_deployment_samples,
    )
    print(summary.to_string(index=False))
    print(
        "Note: upfront_auditing safety is a different quantity "
        "(probability of no successful attack over a simulated deployment of "
        f"{config.upfront_deployment_samples} samples) and is not comparable "
        "to the other protocols' caught/(caught+uncaught) safety."
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    plot_summary(summary).savefig(args.out_dir / "summary.png")
    # Suspicion histogram for a monitored protocol (baselines carry no scores).
    protocol = "trusted_monitoring"
    plot_suspicion(df, protocol).savefig(args.out_dir / f"suspicion_{protocol}.png")
    print(f"Saved plots to {args.out_dir}/")


if __name__ == "__main__":
    main()
