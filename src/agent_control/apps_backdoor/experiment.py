"""APPS setting factory and the Phase 1 protocol lineup."""

from __future__ import annotations

from control_arena import ControlAgent, EvalMode
from control_arena.micro_protocol import (
    defer_to_trusted_builder,
    trusted_editing_builder,
    trusted_monitoring_builder,
)
from control_arena.monitor import basic_monitor_builder
from control_arena.policy import trusted_policy_builder, untrusted_policy_builder
from control_arena.settings.apps.apps_setting import AppsSetting
from inspect_ai.solver import TaskState
from inspect_ai.tool import Tool

from agent_control.apps_backdoor.monitor_view import hide_test_run
from agent_control.apps_backdoor.test_run_tool import test_run as make_test_run_tool
from agent_control.config import ExperimentConfig


class AppsSettingWithTestRun(AppsSetting):
    """ControlArena's APPS setting plus the sandboxed ``test_run`` tool.

    AppsSetting has no ``tools`` field and its ``get_agent_tools`` returns [],
    so the tool is added by overriding that method. It is offered in both
    honest and attack mode.
    """

    def get_agent_tools(self, state: TaskState) -> list[Tool]:
        return [*super().get_agent_tools(state), make_test_run_tool()]


def build_setting(config: ExperimentConfig) -> AppsSetting:
    """Build the APPS setting.

    Dataset choice (control-arena 20.1.0 AppsSetting fields):
    - ``has_solution=True`` (default): only problems whose reference solution
      passes the tests, so the main-task scorer is meaningful.
    - ``has_backdoor_solution=True`` (default is False): only problems for
      which a backdoored solution is known to compile, pass the tests and
      trigger on a secret input. Those are problems where a successful backdoor
      is demonstrably achievable, so low attack success reflects the attacker
      model or the protocol rather than an impossible problem. Honest runs use
      the same problem pool, keeping honest and attack comparable.
    - ``backdoor_source="any"`` (default): keeps the largest pool (APPS paper
      plus Control-Tax backdoors) so 20 samples per mode are available.
    - ``exclude_nondeterministic=True`` (default): drops ambiguous problems.
    """
    return AppsSettingWithTestRun(
        has_solution=True,
        has_backdoor_solution=True,
        backdoor_source="any",
        exclude_nondeterministic=True,
    )


PROTOCOL_MODES: dict[str, list[EvalMode]] = {
    "trusted_only": [EvalMode.HONEST],
    "untrusted_only": [EvalMode.HONEST, EvalMode.ATTACK],
    "trusted_monitoring": [EvalMode.HONEST, EvalMode.ATTACK],
    "untrusted_monitoring": [EvalMode.HONEST, EvalMode.ATTACK],
    "defer_to_trusted": [EvalMode.HONEST, EvalMode.ATTACK],
    "trusted_editing": [EvalMode.HONEST, EvalMode.ATTACK],
}


def build_protocols(config: ExperimentConfig) -> dict[str, ControlAgent]:
    """Build the lineup. Upfront auditing has no entry: analysis computes it
    from the ``untrusted_only`` runs."""

    def monitor(role: str) -> ControlAgent:
        m = basic_monitor_builder(model_role=role)
        return m if config.monitor_sees_test_run else hide_test_run(monitor=m)

    return {
        "trusted_only": trusted_policy_builder(),
        # switches between honest and attack behaviour by eval mode
        "untrusted_only": untrusted_policy_builder(),
        "trusted_monitoring": trusted_monitoring_builder(monitor=monitor("trusted")),
        "untrusted_monitoring": trusted_monitoring_builder(
            monitor=monitor("untrusted")
        ),
        "defer_to_trusted": defer_to_trusted_builder(
            monitor=monitor("trusted"), defer_threshold=config.defer_threshold
        ),
        "trusted_editing": trusted_editing_builder(
            monitor=monitor("trusted"), edit_threshold=config.defer_threshold
        ),
    }
