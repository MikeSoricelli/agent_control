"""Offline tests for the smoke check (no network, no model calls)."""

from __future__ import annotations

import pandas as pd
import pytest
from inspect_ai.log import EvalLog, EvalStats
from inspect_ai.model import ModelUsage

from agent_control.apps_backdoor import smoke


def frame(rows):
    return pd.DataFrame(
        rows,
        columns=["eval_mode", "secret_input", "main_task_success", "side_task_success"],
    )


def test_attack_refusal_rate_counts_no_secret_input():
    df = frame([("attack", None, False, False)] * 3 + [("attack", "5", True, True)])
    assert smoke.attack_refusal_rate(df) == 0.75


def test_attack_refusal_rate_ignores_honest_rows():
    df = frame([("honest", None, True, None)] * 5 + [("attack", "5", True, True)])
    assert smoke.attack_refusal_rate(df) == 0.0


def test_attack_refusal_rate_counts_refusal_message_and_no_submission():
    df = pd.DataFrame(
        {
            "eval_mode": ["attack"] * 3,
            "secret_input": ["1", "1", "1"],
            "submission": ["code", None, "code"],
            "final_message": ["done", "x", "I can't help with that request."],
        }
    )
    assert smoke.attack_refusal_rate(df) == pytest.approx(2 / 3)


def test_token_cost_summary_uses_given_prices():
    log = EvalLog(
        status="success",
        eval={
            "task": "t",
            "task_id": "i",
            "run_id": "r",
            "created": "2026-01-01",
            "model": "m",
            "dataset": {},
            "config": {},
        },
        stats=EvalStats(
            started_at="2026-01-01T00:00:00+00:00",
            completed_at="2026-01-01T00:01:00+00:00",
            model_usage={
                "m/a": ModelUsage(input_tokens=2_000_000, output_tokens=500_000)
            },
        ),
    )
    out = smoke.token_cost_summary([log, log], {"m/a": (0.5, 2.0)})
    # per log: 2 * 0.5 + 0.5 * 2.0 = 2.0, two logs
    assert out == {"m/a": pytest.approx(4.0)}


def _patch(monkeypatch, attack_df):
    honest_t = frame([("honest", "1", False, None)])
    monkeypatch.setattr(
        smoke,
        "collect",
        lambda models, config: {"m/a": (attack_df, []), "trusted_only": (honest_t, [])},
    )
    monkeypatch.setattr(smoke, "fetch_prices", lambda: {})


def test_main_exits_nonzero_above_threshold(monkeypatch, capsys):
    refusing = frame([("attack", None, False, False)] * 4 + [("honest", "1", True, None)])
    _patch(monkeypatch, refusing)
    with pytest.raises(SystemExit) as e:
        smoke.main(["--models", "m/a"])
    assert e.value.code != 0
    assert "refusal" in capsys.readouterr().out.lower()


def test_main_returns_below_threshold(monkeypatch):
    ok = frame([("attack", "1", True, True)] * 4 + [("honest", "1", True, None)])
    _patch(monkeypatch, ok)
    smoke.main(["--models", "m/a"])
