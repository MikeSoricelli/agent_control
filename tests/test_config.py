import pytest

from agent_control.config import ExperimentConfig


def test_defaults_match_spec():
    cfg = ExperimentConfig()
    assert cfg.untrusted_model == "openrouter/deepseek/deepseek-v3.2"
    assert cfg.trusted_model == "openrouter/google/gemini-2.5-flash"
    assert cfg.samples_per_mode == 20
    assert cfg.monitor_sees_test_run is False


def test_to_control_eval_config_maps_fields():
    cfg = ExperimentConfig()
    ec = cfg.to_control_eval_config()
    assert ec.untrusted_model == cfg.untrusted_model
    assert ec.trusted_model == cfg.trusted_model
    assert ec.limit == cfg.samples_per_mode
    assert ec.max_steps == cfg.max_steps


def test_require_api_key_raises_without_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        ExperimentConfig().require_api_key()
