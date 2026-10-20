"""Generic task-grid runner: one Inspect task per (protocol, mode)."""

from __future__ import annotations

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
    return inspect_eval(
        tasks,
        log_dir=str(config.log_dir),
        limit=limit if limit is not None else config.samples_per_mode,
        max_steps=config.max_steps,
    )
