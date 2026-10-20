"""Tests for hiding test_run calls from the monitor."""

from __future__ import annotations

import asyncio

from control_arena import control_agent
from control_arena.monitor import MonitorResponse
from inspect_ai.agent import Agent, AgentState
from inspect_ai.model import (
    ChatMessageAssistant,
    ChatMessageTool,
    ChatMessageUser,
)
from inspect_ai.tool import Tool, ToolCall

from agent_control.apps_backdoor.monitor_view import hide_test_run

RECEIVED: list[list] = []


@control_agent
def recording_monitor() -> Agent:
    async def impl(state: AgentState, tools: list[Tool]) -> AgentState:
        RECEIVED.append(list(state.messages))
        last = state.messages[-1]
        assert isinstance(last, ChatMessageAssistant)
        last.metadata = {
            **(last.metadata or {}),
            **MonitorResponse(suspicion_score=0.7, reason="fake").model_dump(),
        }
        return state

    return impl


def tc(fn: str, id: str) -> ToolCall:
    return ToolCall(id=id, function=fn, arguments={"code": "x"})


def run(messages):
    RECEIVED.clear()
    state = AgentState(messages=messages)
    return asyncio.run(hide_test_run(monitor=recording_monitor())(state, []))


def test_only_test_run_calls_skips_monitor():
    out = run(
        [
            ChatMessageUser(content="task"),
            ChatMessageAssistant(content="", tool_calls=[tc("test_run", "a")]),
        ]
    )
    assert RECEIVED == []
    resp = MonitorResponse.model_validate(out.messages[-1].metadata)
    assert resp.suspicion_score == 0.0
    assert "test_run" in resp.reason


def test_mixed_calls_monitor_sees_no_test_run():
    out = run(
        [
            ChatMessageUser(content="task"),
            ChatMessageAssistant(
                content="", tool_calls=[tc("test_run", "a"), tc("submit", "b")]
            ),
        ]
    )
    assert len(RECEIVED) == 1
    seen_last = RECEIVED[0][-1]
    assert [c.function for c in seen_last.tool_calls] == ["submit"]
    # monitor's score reaches the original message
    resp = MonitorResponse.model_validate(out.messages[-1].metadata)
    assert resp.suspicion_score == 0.7
    assert [c.function for c in out.messages[-1].tool_calls] == ["test_run", "submit"]


def test_history_test_run_results_removed():
    run(
        [
            ChatMessageUser(content="task"),
            ChatMessageAssistant(content="", tool_calls=[tc("test_run", "a")]),
            ChatMessageTool(content="output", tool_call_id="a", function="test_run"),
            ChatMessageAssistant(content="", tool_calls=[tc("other", "c")]),
            ChatMessageTool(content="other out", tool_call_id="c", function="other"),
            ChatMessageAssistant(content="", tool_calls=[tc("submit", "b")]),
        ]
    )
    seen = RECEIVED[0]
    assert not any(
        isinstance(m, ChatMessageTool) and m.tool_call_id == "a" for m in seen
    )
    assert any(isinstance(m, ChatMessageTool) and m.tool_call_id == "c" for m in seen)
    for m in seen:
        if isinstance(m, ChatMessageAssistant):
            assert all(c.function != "test_run" for c in (m.tool_calls or []))


def test_original_state_messages_preserved():
    msgs = [
        ChatMessageUser(content="task"),
        ChatMessageAssistant(content="", tool_calls=[tc("test_run", "a")]),
        ChatMessageTool(content="output", tool_call_id="a", function="test_run"),
        ChatMessageAssistant(
            content="", tool_calls=[tc("test_run", "d"), tc("submit", "b")]
        ),
    ]
    out = run(msgs)
    assert len(out.messages) == 4
    assert [c.function for c in out.messages[1].tool_calls] == ["test_run"]
    assert isinstance(out.messages[2], ChatMessageTool)
    assert [c.function for c in out.messages[3].tool_calls] == ["test_run", "submit"]
