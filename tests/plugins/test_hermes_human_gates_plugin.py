from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
from plugins.hermes_human_gates import (
    _RUNTIME_BY_ADAPTER,
    _send_prompt,
    _shutdown_all,
    _wire_telegram,
    register,
)


def _ctx():
    mgr = PluginManager()
    manifest = PluginManifest(
        name="hermes-human-gates",
        version="1.0.0",
        description="test",
        kind="backend",
    )
    return mgr, PluginContext(manifest=manifest, manager=mgr)


def test_register_queues_scoped_telegram_handler_and_unload(monkeypatch):
    monkeypatch.setenv("HERMES_GATE_BRIDGE_ENABLED", "true")
    mgr, ctx = _ctx()
    register(ctx)
    factories = mgr.get_platform_handler_factories("telegram")
    assert factories == [(_wire_telegram, "hermes-human-gates")]


def test_register_is_inert_when_bridge_disabled(monkeypatch):
    monkeypatch.delenv("HERMES_GATE_BRIDGE_ENABLED", raising=False)
    mgr, ctx = _ctx()
    register(ctx)
    assert mgr.get_platform_handler_factories("telegram") == []


def test_disabled_by_default_does_not_add_handler(monkeypatch):
    monkeypatch.delenv("HERMES_GATE_BRIDGE_ENABLED", raising=False)
    app = MagicMock()
    adapter = MagicMock()

    _wire_telegram(app, adapter)

    app.add_handler.assert_not_called()
    assert _RUNTIME_BY_ADAPTER == {}


@pytest.mark.asyncio
async def test_wired_callback_uses_dedicated_bridge_and_accepts_update(monkeypatch):
    fake = MagicMock()
    fake.start = AsyncMock()
    fake.stop = AsyncMock()
    fake.close = MagicMock()
    fake.handle_callback = AsyncMock(return_value="Hermes gate approved.")

    app = MagicMock()
    adapter = MagicMock()
    adapter._accept_update = MagicMock()

    with patch(
        "plugins.hermes_human_gates.HermesGateBridge.from_env",
        return_value=fake,
    ):
        _wire_telegram(app, adapter)
        await asyncio.sleep(0)

    handler = app.add_handler.call_args.args[0]
    assert handler.pattern.pattern == r"^hg:(?:a|r):"

    query = MagicMock()
    query.data = "hg:a:opaque"
    query.message = MagicMock()
    query.message.chat_id = 202
    query.from_user = MagicMock()
    query.from_user.id = 101
    query.answer = AsyncMock()
    update = MagicMock()
    update.callback_query = query

    await handler.callback(update, None)

    adapter._accept_update.assert_called_once_with()
    fake.handle_callback.assert_awaited_once_with(
        "hg:a:opaque", user_id="101", chat_id="202"
    )
    query.answer.assert_awaited_once()
    assert "approved" in query.answer.await_args.kwargs["text"].lower()

    _shutdown_all()
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    fake.stop.assert_awaited()
    fake.close.assert_called_once()


@pytest.mark.asyncio
async def test_reconnect_reuses_one_bridge_runtime():
    fake = MagicMock()
    fake.start = AsyncMock()
    fake.stop = AsyncMock()
    fake.close = MagicMock()
    fake.handle_callback = AsyncMock(return_value="ok")

    adapter = MagicMock()
    adapter.is_connected = True
    app1 = MagicMock()
    app2 = MagicMock()

    with patch(
        "plugins.hermes_human_gates.HermesGateBridge.from_env",
        return_value=fake,
    ) as factory:
        _wire_telegram(app1, adapter)
        await asyncio.sleep(0)
        _wire_telegram(app2, adapter)
        await asyncio.sleep(0)

    assert factory.call_count == 1
    app1.add_handler.assert_called_once()
    app2.add_handler.assert_called_once()
    assert len(_RUNTIME_BY_ADAPTER) == 1

    _shutdown_all()
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    fake.stop.assert_awaited()
    fake.close.assert_called_once()


@pytest.mark.asyncio
async def test_send_prompt_keeps_compact_callbacks_and_info_mode():
    bot = MagicMock()
    bot.send_message = AsyncMock(return_value=SimpleNamespace(message_id=42))
    adapter = MagicMock()
    adapter._bot = bot
    adapter._link_preview_kwargs.return_value = {}

    wait = {
        "job_id": "job-1",
        "project": "project",
        "gate": "HERMES_PHASE14_TELEGRAM_TEST",
        "source_run_id": "run-1",
        "attempt": 1,
        "summary": "Harmless approval smoke.",
    }

    message_id = await _send_prompt(adapter, "202", wait, "abc123", True)
    assert message_id == "42"
    kwargs = bot.send_message.await_args.kwargs
    buttons = [
        button.callback_data
        for row in kwargs["reply_markup"].inline_keyboard
        for button in row
    ]
    assert buttons == ["hg:a:abc123", "hg:r:abc123"]
    assert all(len(value.encode("utf-8")) <= 64 for value in buttons)

    bot.send_message.reset_mock()
    await _send_prompt(adapter, "202", wait, "abc123", False)
    kwargs = bot.send_message.await_args.kwargs
    assert kwargs["reply_markup"] is None
    assert "Informational only" in kwargs["text"]
