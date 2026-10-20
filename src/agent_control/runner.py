"""Generic task-grid runner: one Inspect task per (protocol, mode)."""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Mapping, Sequence

from control_arena import ControlAgent, EvalMode
from control_arena.eval import get_control_task
from control_arena.scorers import max_suspicion_scorer
from control_arena.settings._setting import Setting
from inspect_ai import Task
from inspect_ai import eval as inspect_eval
from inspect_ai.log import EvalLog

from agent_control.config import ExperimentConfig


def build_tasks(
    setting: Setting,
    protocols: dict[str, ControlAgent],
    modes: Mapping[str, Sequence[EvalMode]],
    config: ExperimentConfig,
) -> list[Task]:
    """Build one task per (protocol, mode) in ``modes[protocol_name]``.

    Task names start with the protocol name. ``max_suspicion_scorer`` is added
    explicitly (get_control_task does not add it) so analysis can read the
    per-sample peak suspicion.
    """
    # get_control_task instantiates the models, so fail fast with a clear message.
    config.require_api_key()
    eval_config = config.to_control_eval_config()
    tasks: list[Task] = []
    for name, protocol in protocols.items():
        for mode in modes[name]:
            tasks.append(
                get_control_task(
                    setting,
                    protocol,
                    mode,
                    eval_config=eval_config,
                    monitoring_scorers=[max_suspicion_scorer()],
                    task_name_prefix=name,
                )
            )
    return tasks


def run_tasks(
    tasks: list[Task], config: ExperimentConfig, limit: int | None = None
) -> list[EvalLog]:
    """Run tasks, failing fast if the API key is missing. ``limit`` caps samples."""
    config.require_api_key()
    # max_steps already reaches the agent loop via control_args() in build_tasks;
    # eval_args() carries ControlArena's fail_on_error/max_tasks/etc. defaults.
    eval_args = {
        **config.to_control_eval_config().eval_args(),
        "limit": limit if limit is not None else config.samples_per_mode,
        "log_dir": str(config.log_dir),
    }
    return inspect_eval(tasks, **eval_args)


def require_docker() -> None:
    """Fail fast with an actionable message when Docker is unreachable."""
    hint = (
        "Docker is required for the APPS sandbox. Start Docker Desktop and wait "
        "until it is running. If `docker` is not found or errors on credentials, "
        "check that the docker CLI is on PATH and that the credential helper in "
        "~/.docker/config.json is installed."
    )
    if shutil.which("docker") is None:
        raise RuntimeError(f"docker CLI not found. {hint}")
    try:
        result = subprocess.run(
            ["docker", "info"], capture_output=True, text=True, timeout=30
        )
    except (subprocess.TimeoutExpired, OSError) as e:
        raise RuntimeError(f"docker did not respond ({e}). {hint}") from e
    if result.returncode != 0:
        detail = (result.stderr or "").strip()
        raise RuntimeError(f"docker daemon unreachable: {detail}. {hint}")
