"""Tests for the DVK Hermes human-gate bridge used by Telegram."""

from __future__ import annotations

import sqlite3
from unittest.mock import AsyncMock

import pytest

from plugins.hermes_human_gates.bridge import (
    HermesGateBridge,
    HermesGateHTTPError,
    HermesGateTransportError,
)


GATE = "HERMES_PHASE14_TELEGRAM_TEST"
WAIT = {
    "job_id": "job-1",
    "project": "test-project",
    "gate": GATE,
    "source_run_id": "run-1",
    "attempt": 1,
    "summary": "Harmless approval smoke.",
}


def make_bridge(tmp_path, *, users=None, chats=None, actionable=None, clock=None):
    send = AsyncMock(return_value="42")
    edit = AsyncMock()
    bridge = HermesGateBridge(
        controller_url="http://127.0.0.1:8787",
        operator_token="operator-secret",
        state_path=tmp_path / "human-gates.db",
        approver_users=set(users if users is not None else {"101"}),
        approver_chats=set(chats if chats is not None else {"202"}),
        actionable_gates=set(actionable if actionable is not None else {GATE}),
        send_prompt=send,
        edit_prompt=edit,
        poll_interval_seconds=1,
        callback_ttl_seconds=60,
        clock=clock,
    )
    return bridge, send, edit


@pytest.mark.asyncio
async def test_poll_persists_short_callback_and_restart_deduplicates(tmp_path):
    bridge, send, _ = make_bridge(tmp_path)
    bridge._request_json = AsyncMock(return_value={"gates": [WAIT]})

    await bridge.poll_once()
    send.assert_awaited_once()
    args = send.await_args.args
    assert args[0] == "202"
    assert args[1] == WAIT
    opaque_id = args[2]
    assert args[3] is True
    assert len(f"hg:a:{opaque_id}".encode("utf-8")) <= 64
    assert len(f"hg:r:{opaque_id}".encode("utf-8")) <= 64
    bridge.close()

    bridge2, send2, _ = make_bridge(tmp_path)
    bridge2._request_json = AsyncMock(return_value={"gates": [WAIT]})
    await bridge2.poll_once()
    send2.assert_not_awaited()
    bridge2.close()


@pytest.mark.asyncio
async def test_non_actionable_gate_is_informational_only(tmp_path):
    bridge, send, _ = make_bridge(tmp_path, actionable=set())
    bridge._request_json = AsyncMock(return_value={"gates": [WAIT]})
    await bridge.poll_once()
    assert send.await_args.args[3] is False
    bridge.close()


@pytest.mark.asyncio
async def test_callback_requires_dedicated_user_and_chat_allowlists(tmp_path):
    bridge, send, edit = make_bridge(tmp_path)
    bridge._request_json = AsyncMock(return_value={"gates": [WAIT]})
    await bridge.poll_once()
    opaque_id = send.await_args.args[2]
    bridge._request_json.reset_mock()

    answer = await bridge.handle_callback(
        f"hg:a:{opaque_id}", user_id="999", chat_id="202"
    )
    assert "not authorized" in answer.lower()
    bridge._request_json.assert_not_awaited()
    edit.assert_not_awaited()

    answer = await bridge.handle_callback(
        f"hg:a:{opaque_id}", user_id="101", chat_id="999"
    )
    assert "not authorized" in answer.lower()
    bridge._request_json.assert_not_awaited()
    bridge.close()


@pytest.mark.asyncio
async def test_approve_sends_bound_source_run_and_resolves_once(tmp_path):
    bridge, send, edit = make_bridge(tmp_path)
    bridge._request_json = AsyncMock(return_value={"gates": [WAIT]})
    await bridge.poll_once()
    opaque_id = send.await_args.args[2]

    bridge._request_json = AsyncMock(
        return_value={
            "decision": "APPROVED",
            "disposition": "QUEUED",
            "source_run_id": "run-1",
        }
    )
    answer = await bridge.handle_callback(
        f"hg:a:{opaque_id}", user_id="101", chat_id="202"
    )
    assert "approved" in answer.lower()
    bridge._request_json.assert_awaited_once_with(
        "POST",
        "/v1/gates/approve",
        {
            "job_id": "job-1",
            "gate": GATE,
            "source_run_id": "run-1",
            "actor": "telegram:101",
        },
    )
    edit.assert_awaited_once()
    assert "APPROVED" in edit.await_args.args[3]

    bridge._request_json.reset_mock()
    second = await bridge.handle_callback(
        f"hg:a:{opaque_id}", user_id="101", chat_id="202"
    )
    assert "already" in second.lower()
    bridge._request_json.assert_not_awaited()
    bridge.close()


@pytest.mark.asyncio
async def test_network_failure_returns_prompt_to_pending(tmp_path):
    bridge, send, edit = make_bridge(tmp_path)
    bridge._request_json = AsyncMock(return_value={"gates": [WAIT]})
    await bridge.poll_once()
    opaque_id = send.await_args.args[2]

    bridge._request_json = AsyncMock(
        side_effect=HermesGateTransportError("controller unreachable")
    )
    answer = await bridge.handle_callback(
        f"hg:r:{opaque_id}", user_id="101", chat_id="202"
    )
    assert "retry" in answer.lower()
    edit.assert_not_awaited()

    row = (
        sqlite3
        .connect(tmp_path / "human-gates.db")
        .execute("SELECT state FROM prompts WHERE opaque_id=?", (opaque_id,))
        .fetchone()
    )
    assert row[0] == "PENDING"
    bridge.close()


@pytest.mark.asyncio
async def test_conflict_marks_prompt_stale_after_reconciliation(tmp_path):
    bridge, send, edit = make_bridge(tmp_path)
    bridge._request_json = AsyncMock(return_value={"gates": [WAIT]})
    await bridge.poll_once()
    opaque_id = send.await_args.args[2]

    bridge._request_json = AsyncMock(
        side_effect=[
            HermesGateHTTPError(409, "gate source run does not match"),
            {
                "job_id": "job-1",
                "state": "QUEUED",
                "attempt": 1,
                "source_run_id": None,
                "gate": {"waiting_for": None, "approved": [GATE], "rejected": []},
                "decisions": [],
            },
        ]
    )
    answer = await bridge.handle_callback(
        f"hg:a:{opaque_id}", user_id="101", chat_id="202"
    )
    assert "no longer pending" in answer.lower()
    assert bridge._request_json.await_count == 2
    assert bridge._request_json.await_args_list[1].args == (
        "GET",
        "/v1/gates/job-1",
        None,
    )
    assert "STALE" in edit.await_args.args[3]
    bridge.close()


@pytest.mark.asyncio
async def test_expired_callback_is_not_sent_to_controller(tmp_path):
    now = [1_000]
    bridge, send, edit = make_bridge(tmp_path, clock=lambda: now[0])
    bridge._request_json = AsyncMock(return_value={"gates": [WAIT]})
    await bridge.poll_once()
    opaque_id = send.await_args.args[2]
    bridge._request_json.reset_mock()

    now[0] = 2_000
    answer = await bridge.handle_callback(
        f"hg:a:{opaque_id}", user_id="101", chat_id="202"
    )
    assert "expired" in answer.lower()
    bridge._request_json.assert_not_awaited()
    assert "EXPIRED" in edit.await_args.args[3]
    bridge.close()


@pytest.mark.asyncio
async def test_global_gateway_allow_all_does_not_enable_gate_actions(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_GATE_BRIDGE_ENABLED", "true")
    monkeypatch.setenv("HERMES_OPERATOR_TOKEN", "operator-secret")
    monkeypatch.setenv("HERMES_GATE_APPROVER_CHATS", "202")
    monkeypatch.delenv("HERMES_GATE_APPROVER_USERS", raising=False)
    monkeypatch.setenv("HERMES_GATE_ACTIONABLE_GATES", GATE)
    monkeypatch.setenv("GATEWAY_ALLOW_ALL_USERS", "true")
    monkeypatch.setenv("HERMES_GATE_STATE_DB", str(tmp_path / "env-gates.db"))

    send = AsyncMock(return_value="42")
    edit = AsyncMock()
    bridge = HermesGateBridge.from_env(send_prompt=send, edit_prompt=edit)
    assert bridge is not None
    bridge._request_json = AsyncMock(return_value={"gates": [WAIT]})

    await bridge.poll_once()

    assert send.await_args.args[3] is False
    answer = await bridge.handle_callback("hg:a:anything", user_id="101", chat_id="202")
    assert "not authorized" in answer.lower()
    bridge.close()


@pytest.mark.asyncio
async def test_operator_401_marks_bridge_auth_failed(tmp_path):
    bridge, send, edit = make_bridge(tmp_path)
    bridge._request_json = AsyncMock(return_value={"gates": [WAIT]})
    await bridge.poll_once()
    opaque_id = send.await_args.args[2]

    bridge._request_json = AsyncMock(
        side_effect=HermesGateHTTPError(401, "unauthorized")
    )
    answer = await bridge.handle_callback(
        f"hg:a:{opaque_id}", user_id="101", chat_id="202"
    )

    assert "authorization failed" in answer.lower()
    assert bridge._auth_failed is True
    assert "AUTH_FAILED" in edit.await_args.args[3]
    bridge.close()


def test_restart_recovers_in_flight_callback_to_pending(tmp_path):
    bridge, send, _ = make_bridge(tmp_path)
    bridge.db.execute(
        """INSERT INTO prompts
           (opaque_id, job_id, gate, source_run_id, chat_id, message_id,
            state, created_at, expires_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        ("recover-me", "job-1", GATE, "run-1", "202", "42", "IN_FLIGHT", 1, 999999),
    )
    bridge.close()

    bridge2, _, _ = make_bridge(tmp_path)
    row = bridge2.db.execute(
        "SELECT state FROM prompts WHERE opaque_id='recover-me'"
    ).fetchone()
    assert row["state"] == "PENDING"
    bridge2.close()


@pytest.mark.asyncio
async def test_restart_revokes_previously_actionable_gate(tmp_path):
    bridge, send, _ = make_bridge(tmp_path)
    bridge._request_json = AsyncMock(return_value={"gates": [WAIT]})
    await bridge.poll_once()
    opaque_id = send.await_args.args[2]
    bridge.close()
    restarted, _, edit = make_bridge(tmp_path, actionable=set())
    restarted._request_json = AsyncMock(return_value={"decision": "APPROVED"})
    answer = await restarted.handle_callback(
        f"hg:a:{opaque_id}", user_id="101", chat_id="202"
    )
    assert "informational" in answer.lower()
    restarted._request_json.assert_not_awaited()
    assert restarted.db.execute("SELECT state FROM prompts").fetchone()[0] == "INFO"
    restarted.close()


@pytest.mark.asyncio
async def test_auth_failure_blocks_other_pending_callbacks(tmp_path):
    bridge, send, _ = make_bridge(tmp_path)
    bridge._request_json = AsyncMock(
        return_value={
            "gates": [WAIT, {**WAIT, "job_id": "job-2", "source_run_id": "run-2"}]
        }
    )
    await bridge.poll_once()
    first, second = [call.args[2] for call in send.await_args_list]
    bridge._request_json = AsyncMock(
        side_effect=HermesGateHTTPError(401, "unauthorized")
    )
    await bridge.handle_callback(f"hg:a:{first}", user_id="101", chat_id="202")
    bridge._request_json.reset_mock()
    answer = await bridge.handle_callback(
        f"hg:a:{second}", user_id="101", chat_id="202"
    )
    assert "authorization" in answer.lower()
    bridge._request_json.assert_not_awaited()
    bridge.close()
