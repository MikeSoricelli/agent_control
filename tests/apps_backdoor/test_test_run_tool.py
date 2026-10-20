"""Tests for the apps_backdoor test-run tool."""

from __future__ import annotations

import time

import pytest
from inspect_ai import Task, eval
from inspect_ai.dataset import Sample
from inspect_ai.scorer import Score, Target, accuracy, scorer
from inspect_ai.solver import TaskState, solver
from inspect_ai.util import store

from agent_control.apps_backdoor.test_run_tool import (
    TEST_RUN_TOOL_NAME,
    format_run_result,
)
from agent_control.apps_backdoor.test_run_tool import test_run as make_test_run_tool


def test_tool_name():
    assert TEST_RUN_TOOL_NAME == "test_run"


def test_format_run_result_truncates_to_max_chars():
    out = format_run_result("a" * 10_000, "b" * 10_000, 0, False, max_chars=100)
    assert len(out) < 600
    assert "truncated" in out
    assert "a" * 100 in out
    assert "a" * 101 not in out


def test_format_run_result_reports_timeout_and_nonzero_exit():
    out = format_run_result("", "", -1, True)
    assert "timed out" in out
    out = format_run_result("", "Traceback boom", 3, False)
    assert "3" in out
    assert "Traceback boom" in out


def test_format_run_result_success_has_stdout():
    out = format_run_result("hello", "", 0, False)
    assert "hello" in out
    assert "timed out" not in out


@solver
def _call_tool(code: str, input_str: str, timeout_s: int):
    async def solve(state: TaskState, generate):
        tool = make_test_run_tool(timeout_s=timeout_s)
        start = time.monotonic()
        result = await tool(code=code, input=input_str)
        store().set("result", str(result))
        store().set("elapsed", time.monotonic() - start)
        return state

    return solve


@scorer(metrics=[accuracy()])
def _noop():
    async def score(state: TaskState, target: Target):
        return Score(value=1.0)

    return score


def _run(code: str, input_str: str, timeout_s: int = 10):
    task = Task(
        dataset=[Sample(input="x")],
        solver=_call_tool(code, input_str, timeout_s),
        scorer=_noop(),
        sandbox="docker",
    )
    log = eval(task, model="mockllm/model", display="none")[0]
    assert log.status == "success", log.error
    sample = log.samples[0]
    return sample.store["result"], sample.store["elapsed"]


@pytest.mark.docker
def test_test_run_executes_code_in_sandbox():
    result, _ = _run("print(input())", "hello world\n")
    assert "hello world" in result


@pytest.mark.docker
def test_test_run_infinite_loop_times_out():
    result, elapsed = _run("while True: pass", "", timeout_s=3)
    assert "timed out" in result
    assert elapsed < 3 + 15


@pytest.mark.docker
def test_test_run_crash_and_huge_output_are_bounded():
    result, _ = _run("raise ValueError('kaboom')", "")
    assert "kaboom" in result
    assert "exit code" in result
    result, _ = _run("print('x' * 5_000_000)", "")
    assert len(result) < 5000
