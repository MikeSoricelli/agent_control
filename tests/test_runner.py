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


def _mock_cfg(tmp_path):
    return ExperimentConfig(
        untrusted_model="mockllm/model",
        trusted_model="mockllm/model",
        log_dir=tmp_path / "logs",
        samples_per_mode=1,
        max_steps=2,
    )


def test_run_tasks_real_inspect_path_with_mockllm(tmp_path):
    """No patching of inspect_eval: catches kwargs Inspect would reject."""
    from control_arena.policy import trusted_policy_builder

    cfg = _mock_cfg(tmp_path)
    tasks = build_tasks(
        FakeSetting(),
        {"trusted_only": trusted_policy_builder()},
        {"trusted_only": [EvalMode.HONEST]},
        cfg,
    )
    logs = run_tasks(tasks, cfg)
    assert len(logs) == 1
    assert logs[0].status == "success"
    assert list((tmp_path / "logs").glob("*.eval"))


def test_run_tasks_forwards_control_eval_args(tmp_path, protocols, modes):
    cfg = _mock_cfg(tmp_path)
    tasks = build_tasks(FakeSetting(), protocols, modes, cfg)
    with patch("agent_control.runner.inspect_eval", return_value=[]) as mock_eval:
        run_tasks(tasks, cfg, limit=2)
    _, kwargs = mock_eval.call_args
    assert kwargs["fail_on_error"] is False  # ControlArena default, not Inspect's True
    assert kwargs["max_tasks"] == cfg.to_control_eval_config().max_tasks
    assert kwargs["limit"] == 2
    assert "max_steps" not in kwargs


def test_require_docker_ok(monkeypatch):
    from agent_control.runner import require_docker

    monkeypatch.setattr("agent_control.runner.shutil.which", lambda _: "/usr/bin/docker")
    with patch("agent_control.runner.subprocess.run") as run:
        run.return_value.returncode = 0
        require_docker()


def test_require_docker_missing_cli(monkeypatch):
    from agent_control.runner import require_docker

    monkeypatch.setattr("agent_control.runner.shutil.which", lambda _: None)
    with pytest.raises(RuntimeError, match="Docker Desktop"):
        require_docker()


def test_require_docker_daemon_down(monkeypatch):
    from agent_control.runner import require_docker

    monkeypatch.setattr("agent_control.runner.shutil.which", lambda _: "/usr/bin/docker")
    with patch("agent_control.runner.subprocess.run") as run:
        run.return_value.returncode = 1
        run.return_value.stderr = "Cannot connect to the Docker daemon"
        with pytest.raises(RuntimeError, match="credential"):
            require_docker()


def test_require_docker_timeout(monkeypatch):
    import subprocess

    from agent_control.runner import require_docker

    monkeypatch.setattr("agent_control.runner.shutil.which", lambda _: "/usr/bin/docker")
    with patch(
        "agent_control.runner.subprocess.run",
        side_effect=subprocess.TimeoutExpired("docker", 10),
    ):
        with pytest.raises(RuntimeError, match="Docker Desktop"):
            require_docker()


def test_build_tasks_requires_api_key(monkeypatch, protocols, modes):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with patch("agent_control.config.dotenv.load_dotenv"):
        with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
            build_tasks(FakeSetting(), protocols, modes, ExperimentConfig())
