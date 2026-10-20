"""Offline tests for the generic task-grid runner."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from control_arena import EvalMode
from control_arena.micro_protocol import trusted_monitoring_builder
from control_arena.policy import honest_policy_builder
from control_arena.settings._setting import Setting
from inspect_ai.dataset import Sample

from agent_control.config import ExperimentConfig
from agent_control.runner import build_tasks, run_tasks


class FakeSetting(Setting):
    @classmethod
    def get_name(cls) -> str:
        return "fake"

    def get_dataset(self, eval_mode):
        return [Sample(input="x", metadata={})]

    def get_agent_tools(self, state):
        return []

    def get_scorers(self):
        return []

    def get_sandbox_config(self):
        return None


@pytest.fixture(autouse=True)
def dummy_key(monkeypatch):
    # get_control_task instantiates the models, which needs a key present.
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy")


@pytest.fixture
def protocols():
    return {
        "trusted_only": honest_policy_builder(),
        "trusted_monitoring": trusted_monitoring_builder(),
    }


@pytest.fixture
def modes():
    return {
        "trusted_only": [EvalMode.HONEST],
        "trusted_monitoring": [EvalMode.HONEST, EvalMode.ATTACK],
    }


def test_build_tasks_one_per_protocol_mode(protocols, modes):
    tasks = build_tasks(FakeSetting(), protocols, modes, ExperimentConfig())
    assert len(tasks) == 3


def test_task_names_prefixed_with_protocol(protocols, modes):
    tasks = build_tasks(FakeSetting(), protocols, modes, ExperimentConfig())
    names = [t.name for t in tasks]
    assert sum(n.startswith("trusted_only") for n in names) == 1
    assert sum(n.startswith("trusted_monitoring") for n in names) == 2


def test_run_tasks_requires_api_key(monkeypatch, protocols, modes):
    cfg = ExperimentConfig()
    tasks = build_tasks(FakeSetting(), protocols, modes, cfg)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with (
        patch("agent_control.config.dotenv.load_dotenv"),
        patch("agent_control.runner.inspect_eval") as mock_eval,
    ):
        with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
            run_tasks(tasks, cfg)
        mock_eval.assert_not_called()


def test_run_tasks_passes_log_dir_and_limit(monkeypatch, protocols, modes, tmp_path):
    cfg = ExperimentConfig(log_dir=tmp_path)
    tasks = build_tasks(FakeSetting(), protocols, modes, cfg)
    with patch("agent_control.runner.inspect_eval", return_value=[]) as mock_eval:
        run_tasks(tasks, cfg, limit=2)
    _, kwargs = mock_eval.call_args
    assert kwargs["log_dir"] == str(tmp_path)
    assert kwargs["limit"] == 2


def test_build_tasks_requires_api_key(monkeypatch, protocols, modes):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with patch("agent_control.config.dotenv.load_dotenv"):
        with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
            build_tasks(FakeSetting(), protocols, modes, ExperimentConfig())
