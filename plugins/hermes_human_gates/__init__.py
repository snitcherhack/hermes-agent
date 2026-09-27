"""Bundled DVK Hermes human-gates bridge for Telegram.

The Hermes Controller remains the sole authority for pending gates and decisions.
This plugin only owns Telegram routing state and callback transport.
"""

from __future__ import annotations

import asyncio
import html
import logging
import os
from functools import partial
from typing import Any

from .bridge import HermesGateBridge

logger = logging.getLogger(__name__)

_RUNTIME_BY_ADAPTER: dict[int, dict[str, Any]] = {}


def _gate_message(
    wait: dict[str, Any],
    *,
    informational: bool = False,
    status_text: str | None = None,
) -> str:
    project = str(wait.get("project") or "").strip()
    gate = str(wait.get("gate") or "").strip()
    job_id = str(wait.get("job_id") or "").strip()
    attempt = wait.get("attempt")
    summary = str(wait.get("summary") or "").strip()

    lines = ["🛡️ <b>Hermes human gate</b>", ""]
    if project:
        lines.append(f"Project: <code>{html.escape(project)}</code>")
    lines.append(f"Gate: <code>{html.escape(gate)}</code>")
    lines.append(f"Job: <code>{html.escape(job_id)}</code>")
    if attempt:
        try:
            lines.append(f"Attempt: {int(attempt)}")
        except (TypeError, ValueError):
            pass
    if summary:
        lines.extend(["", html.escape(summary[:1800])])
    if informational:
        lines.extend([
            "",
            "ℹ️ Informational only — this gate cannot be resolved from Telegram.",
        ])
    if status_text:
        lines.extend(["", f"<b>{html.escape(status_text)}</b>"])
    return "\n".join(lines)


def _link_preview_kwargs(adapter: Any) -> dict[str, Any]:
    helper = getattr(adapter, "_link_preview_kwargs", None)
    if not callable(helper):
        return {}
    try:
        value = helper()
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


async def _send_prompt(
    adapter: Any,
    chat_id: str,
    wait: dict[str, Any],
    opaque_id: str,
    actionable: bool,
) -> str:
    bot = getattr(adapter, "_bot", None)
    if bot is None:
        raise RuntimeError("Telegram bot is not connected")

    from telegram import InlineKeyboardButton, InlineKeyboardMarkup
    from telegram.constants import ParseMode

    reply_markup = None
    if actionable:
        approve = f"hg:a:{opaque_id}"
        reject = f"hg:r:{opaque_id}"
        if len(approve.encode("utf-8")) > 64 or len(reject.encode("utf-8")) > 64:
            raise ValueError("Hermes gate callback_data exceeds Telegram limit")
        reply_markup = InlineKeyboardMarkup([
            [
                InlineKeyboardButton("✅ Approve", callback_data=approve),
                InlineKeyboardButton("❌ Reject", callback_data=reject),
            ]
        ])

    message = await bot.send_message(
        chat_id=int(chat_id),
        text=_gate_message(wait, informational=not actionable),
        parse_mode=ParseMode.HTML,
        reply_markup=reply_markup,
        **_link_preview_kwargs(adapter),
    )
    return str(message.message_id)


async def _edit_prompt(
    adapter: Any,
    chat_id: str,
    message_id: str,
    wait: dict[str, Any],
    status_text: str,
) -> None:
    bot = getattr(adapter, "_bot", None)
    if bot is None:
        return

    from telegram.constants import ParseMode

    await bot.edit_message_text(
        chat_id=int(chat_id),
        message_id=int(message_id),
        text=_gate_message(wait, status_text=status_text),
        parse_mode=ParseMode.HTML,
        reply_markup=None,
        **_link_preview_kwargs(adapter),
    )


async def _supervise_bridge(bridge: HermesGateBridge, adapter: Any) -> None:
    while not bool(getattr(adapter, "is_connected", False)):
        await asyncio.sleep(0.1)
    await bridge.start()
    try:
        await asyncio.Event().wait()
    finally:
        try:
            await bridge.stop()
        finally:
            bridge.close()


def _runtime_for(adapter: Any) -> dict[str, Any] | None:
    key = id(adapter)
    existing = _RUNTIME_BY_ADAPTER.get(key)
    if existing is not None:
        return existing

    bridge = HermesGateBridge.from_env(
        send_prompt=partial(_send_prompt, adapter),
        edit_prompt=partial(_edit_prompt, adapter),
    )
    if bridge is None:
        return None

    loop = asyncio.get_running_loop()
    supervisor = loop.create_task(
        _supervise_bridge(bridge, adapter),
        name="hermes-human-gates-plugin",
    )
    runtime = {"bridge": bridge, "supervisor": supervisor}
    _RUNTIME_BY_ADAPTER[key] = runtime

    def _done(task: asyncio.Task) -> None:
        current = _RUNTIME_BY_ADAPTER.get(key)
        if current is runtime:
            _RUNTIME_BY_ADAPTER.pop(key, None)
        if not task.cancelled():
            try:
                exc = task.exception()
            except Exception:
                exc = None
            if exc is not None:
                logger.error("Hermes human-gate plugin supervisor failed: %s", exc)

    supervisor.add_done_callback(_done)
    logger.info("Hermes human-gate Telegram bridge enabled")
    return runtime


def _wire_telegram(application: Any, adapter: Any) -> None:
    """Wire the scoped Telegram callback and start one bridge runtime per adapter."""

    runtime = _runtime_for(adapter)
    if runtime is None:
        return

    from telegram.ext import CallbackQueryHandler

    async def _on_gate_callback(update, _context) -> None:
        query = getattr(update, "callback_query", None)
        data = str(getattr(query, "data", "") or "") if query is not None else ""
        if query is None or not data:
            return

        accept = getattr(adapter, "_accept_update", None)
        if callable(accept):
            accept()

        message = getattr(query, "message", None)
        chat_id = getattr(message, "chat_id", None)
        user = getattr(query, "from_user", None)
        user_id = getattr(user, "id", None)
        if chat_id is None or user_id is None:
            await query.answer(text="This Hermes gate action is unavailable.")
            return

        answer = await runtime["bridge"].handle_callback(
            data,
            user_id=str(user_id),
            chat_id=str(chat_id),
        )
        await query.answer(text=str(answer)[:180])

    application.add_handler(
        CallbackQueryHandler(_on_gate_callback, pattern=r"^hg:(?:a|r):")
    )


def _shutdown_all() -> None:
    runtimes = list(_RUNTIME_BY_ADAPTER.values())
    _RUNTIME_BY_ADAPTER.clear()
    for runtime in runtimes:
        task = runtime.get("supervisor")
        if isinstance(task, asyncio.Task) and not task.done():
            task.cancel()


def register(ctx) -> None:
    enabled = os.getenv("HERMES_GATE_BRIDGE_ENABLED", "").strip().lower()
    if enabled not in {"1", "true", "yes", "on"}:
        return
    ctx.register_platform_handler("telegram", _wire_telegram)
    ctx.on_unload(_shutdown_all)
