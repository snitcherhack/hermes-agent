"""Bridge between DVK Hermes human gates and messaging adapters.

The Controller remains authoritative. This module stores only notification
routing state; it never reads Controller SQLite and never persists credentials.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import secrets
import sqlite3
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)

SendPrompt = Callable[[str, dict[str, Any], str, bool], Awaitable[str]]
EditPrompt = Callable[[str, str, dict[str, Any], str], Awaitable[None]]


class HermesGateHTTPError(RuntimeError):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = int(status)


class HermesGateTransportError(RuntimeError):
    pass


def _csv_set(value: str | None) -> set[str]:
    return {part.strip() for part in str(value or "").split(",") if part.strip()}


def _env_enabled(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


class HermesGateBridge:
    """Poll pending Controller gates and map Telegram callbacks to decisions."""

    ACTIVE_STATES = ("PENDING", "IN_FLIGHT", "INFO")

    def __init__(
        self,
        *,
        controller_url: str,
        operator_token: str,
        state_path: str | Path,
        approver_users: set[str],
        approver_chats: set[str],
        actionable_gates: set[str],
        send_prompt: SendPrompt,
        edit_prompt: EditPrompt,
        poll_interval_seconds: float = 5.0,
        callback_ttl_seconds: int = 86_400,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if not controller_url:
            raise ValueError("Hermes gate controller URL is required")
        if not operator_token:
            raise ValueError("Hermes operator token is required")
        if not approver_chats:
            raise ValueError("at least one Hermes gate approver chat is required")
        self.controller_url = controller_url.rstrip("/")
        self._operator_token = operator_token
        self.state_path = Path(state_path)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.approver_users = {str(item) for item in approver_users}
        self.approver_chats = {str(item) for item in approver_chats}
        self.actionable_gates = {str(item) for item in actionable_gates}
        self.send_prompt = send_prompt
        self.edit_prompt = edit_prompt
        self.poll_interval_seconds = max(1.0, float(poll_interval_seconds))
        self.callback_ttl_seconds = max(60, int(callback_ttl_seconds))
        self.clock = clock or time.time
        self.db = sqlite3.connect(self.state_path, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode = WAL")
        self._create_schema()
        # A process crash can leave a callback between local claim and HTTP result.
        # Re-open it for retry; Controller source_run_id + idempotency remain authoritative.
        self.db.execute("UPDATE prompts SET state='PENDING' WHERE state='IN_FLIGHT'")
        try:
            os.chmod(self.state_path, 0o600)
        except OSError:
            pass
        self._poll_task: asyncio.Task | None = None
        self._auth_failed = False

    @classmethod
    def from_env(
        cls,
        *,
        send_prompt: SendPrompt,
        edit_prompt: EditPrompt,
    ) -> "HermesGateBridge | None":
        if not _env_enabled("HERMES_GATE_BRIDGE_ENABLED"):
            return None
        home = Path(os.getenv("HERMES_HOME", str(Path.home() / ".hermes")))
        token = os.getenv("HERMES_OPERATOR_TOKEN", "").strip()
        chats = _csv_set(os.getenv("HERMES_GATE_APPROVER_CHATS"))
        users = _csv_set(os.getenv("HERMES_GATE_APPROVER_USERS"))
        actionable = _csv_set(os.getenv("HERMES_GATE_ACTIONABLE_GATES"))
        state_path = Path(
            os.getenv("HERMES_GATE_STATE_DB", str(home / "human-gates-telegram.db"))
        )
        return cls(
            controller_url=os.getenv(
                "HERMES_GATE_CONTROLLER_URL", "http://127.0.0.1:8787"
            ).strip(),
            operator_token=token,
            state_path=state_path,
            approver_users=users,
            approver_chats=chats,
            actionable_gates=actionable,
            send_prompt=send_prompt,
            edit_prompt=edit_prompt,
            poll_interval_seconds=float(
                os.getenv("HERMES_GATE_POLL_INTERVAL_SECONDS", "5")
            ),
            callback_ttl_seconds=int(
                os.getenv("HERMES_GATE_CALLBACK_TTL_SECONDS", "86400")
            ),
        )

    def _create_schema(self) -> None:
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS prompts (
              opaque_id TEXT PRIMARY KEY,
              job_id TEXT NOT NULL,
              gate TEXT NOT NULL,
              source_run_id TEXT NOT NULL,
              chat_id TEXT NOT NULL,
              message_id TEXT,
              state TEXT NOT NULL,
              created_at INTEGER NOT NULL,
              expires_at INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS prompts_source_chat
              ON prompts(source_run_id, chat_id, created_at);
            """
        )

    def _now(self) -> int:
        return int(self.clock())

    async def start(self) -> None:
        if self._poll_task is None or self._poll_task.done():
            self._poll_task = asyncio.create_task(
                self._poll_loop(), name="hermes-human-gates"
            )

    async def stop(self) -> None:
        task = self._poll_task
        self._poll_task = None
        if task and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    def close(self) -> None:
        self.db.close()

    async def _poll_loop(self) -> None:
        while not self._auth_failed:
            try:
                await self.poll_once()
            except HermesGateHTTPError as exc:
                if exc.status == 401:
                    self._auth_failed = True
                    logger.error(
                        "Hermes human-gate bridge operator authentication failed; polling stopped"
                    )
                    return
                logger.warning(
                    "Hermes human-gate bridge Controller HTTP %s: %s",
                    exc.status,
                    exc,
                )
            except HermesGateTransportError as exc:
                logger.warning("Hermes human-gate bridge transport error: %s", exc)
            except Exception:
                logger.exception("Hermes human-gate bridge poll failed")
            await asyncio.sleep(self.poll_interval_seconds)

    async def poll_once(self) -> None:
        if self._auth_failed:
            return
        payload = await self._request_json("GET", "/v1/gates/pending", None)
        gates = payload.get("gates") if isinstance(payload, dict) else None
        if not isinstance(gates, list):
            raise HermesGateTransportError("invalid pending-gates response")

        pending_sources = {
            str(item.get("source_run_id"))
            for item in gates
            if isinstance(item, dict) and item.get("source_run_id")
        }
        await self._reconcile_absent(pending_sources)

        now = self._now()
        for wait in gates:
            if not self._valid_wait(wait):
                logger.warning("Ignoring malformed Hermes pending gate entry")
                continue
            source_run_id = str(wait["source_run_id"])
            gate = str(wait["gate"])
            for chat_id in sorted(self.approver_chats):
                existing = self._latest_active(source_run_id, chat_id)
                if existing is not None:
                    if (
                        existing["state"] == "PENDING"
                        and int(existing["expires_at"]) <= now
                    ):
                        self._set_state(existing["opaque_id"], "EXPIRED")
                        await self._safe_edit(
                            existing, wait, "EXPIRED — request refreshed"
                        )
                    else:
                        continue

                actionable = (
                    gate in self.actionable_gates
                    and bool(self.approver_users)
                    and chat_id in self.approver_chats
                )
                opaque_id = secrets.token_urlsafe(12)
                message_id = await self.send_prompt(
                    chat_id, wait, opaque_id, actionable
                )
                self.db.execute(
                    """INSERT INTO prompts
                       (opaque_id, job_id, gate, source_run_id, chat_id, message_id,
                        state, created_at, expires_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        opaque_id,
                        wait["job_id"],
                        gate,
                        source_run_id,
                        chat_id,
                        str(message_id),
                        "PENDING" if actionable else "INFO",
                        now,
                        now + self.callback_ttl_seconds,
                    ),
                )

    @staticmethod
    def _valid_wait(wait: Any) -> bool:
        if not isinstance(wait, dict):
            return False
        required = ("job_id", "project", "gate", "source_run_id", "attempt", "summary")
        return all(key in wait for key in required) and all(
            isinstance(wait[key], str)
            for key in ("job_id", "project", "gate", "source_run_id", "summary")
        )

    def _latest_active(self, source_run_id: str, chat_id: str) -> sqlite3.Row | None:
        placeholders = ",".join("?" for _ in self.ACTIVE_STATES)
        return self.db.execute(
            f"""SELECT * FROM prompts
                WHERE source_run_id=? AND chat_id=? AND state IN ({placeholders})
                ORDER BY created_at DESC LIMIT 1""",
            (source_run_id, chat_id, *self.ACTIVE_STATES),
        ).fetchone()

    async def _reconcile_absent(self, pending_sources: set[str]) -> None:
        placeholders = ",".join("?" for _ in self.ACTIVE_STATES)
        rows = self.db.execute(
            f"SELECT * FROM prompts WHERE state IN ({placeholders})",
            self.ACTIVE_STATES,
        ).fetchall()
        for row in rows:
            if row["source_run_id"] in pending_sources:
                continue
            self._set_state(row["opaque_id"], "STALE")
            wait = {
                "job_id": row["job_id"],
                "project": "",
                "gate": row["gate"],
                "source_run_id": row["source_run_id"],
                "attempt": 0,
                "summary": "",
            }
            await self._safe_edit(row, wait, "STALE — gate no longer pending")

    async def handle_callback(
        self,
        data: str,
        *,
        user_id: str,
        chat_id: str,
    ) -> str:
        user_id = str(user_id or "")
        chat_id = str(chat_id or "")
        if user_id not in self.approver_users or chat_id not in self.approver_chats:
            return "You are not authorized to resolve Hermes human gates."

        if self._auth_failed:
            return "Hermes operator authorization is unavailable."

        parts = str(data or "").split(":", 2)
        if len(parts) != 3 or parts[0] != "hg" or parts[1] not in {"a", "r"}:
            return "Invalid Hermes gate action."
        action, opaque_id = parts[1], parts[2]
        row = self.db.execute(
            "SELECT * FROM prompts WHERE opaque_id=?", (opaque_id,)
        ).fetchone()
        if row is None or row["chat_id"] != chat_id:
            return "This Hermes gate action is unavailable."

        state = row["state"]
        if state == "RESOLVED":
            return "This Hermes gate has already been resolved."
        if state == "STALE":
            return "This Hermes gate is no longer pending."
        if state == "EXPIRED":
            return "This Hermes gate button has expired."
        if state == "AUTH_FAILED":
            return "Hermes operator authorization is unavailable."
        if state == "INFO":
            return "This Hermes gate is informational only."
        if state == "IN_FLIGHT":
            return "This Hermes gate decision is already being processed."
        if state != "PENDING":
            return "This Hermes gate action is unavailable."

        wait = {
            "job_id": row["job_id"],
            "project": "",
            "gate": row["gate"],
            "source_run_id": row["source_run_id"],
            "attempt": 0,
            "summary": "",
        }
        # Persisted buttons never override the currently configured gate policy.
        if row["gate"] not in self.actionable_gates:
            self._set_state(opaque_id, "INFO")
            await self._safe_edit(row, wait, "INFO — gate is no longer actionable")
            return "This Hermes gate is informational only."

        if int(row["expires_at"]) <= self._now():
            self._set_state(opaque_id, "EXPIRED")
            await self._safe_edit(row, wait, "EXPIRED — request a fresh prompt")
            return "This Hermes gate button has expired."

        changed = self.db.execute(
            "UPDATE prompts SET state='IN_FLIGHT' WHERE opaque_id=? AND state='PENDING'",
            (opaque_id,),
        ).rowcount
        if not changed:
            return "This Hermes gate decision is already being processed."

        endpoint = "/v1/gates/approve" if action == "a" else "/v1/gates/reject"
        decision = "APPROVED" if action == "a" else "REJECTED"
        try:
            result = await self._request_json(
                "POST",
                endpoint,
                {
                    "job_id": row["job_id"],
                    "gate": row["gate"],
                    "source_run_id": row["source_run_id"],
                    "actor": f"telegram:{user_id}",
                },
            )
        except HermesGateHTTPError as exc:
            if exc.status == 409:
                try:
                    await self._request_json("GET", f"/v1/gates/{row['job_id']}", None)
                except (HermesGateHTTPError, HermesGateTransportError):
                    pass
                self._set_state(opaque_id, "STALE")
                await self._safe_edit(row, wait, "STALE — gate no longer pending")
                return "This Hermes gate is no longer pending."
            if exc.status == 401:
                self._auth_failed = True
                self._set_state(opaque_id, "AUTH_FAILED")
                await self._safe_edit(
                    row, wait, "AUTH_FAILED — operator authentication rejected"
                )
                return "Hermes operator authorization failed."
            self._set_state(opaque_id, "PENDING")
            return "Hermes Controller rejected the request; retry later."
        except HermesGateTransportError:
            self._set_state(opaque_id, "PENDING")
            return "Hermes Controller network error; retry this button."

        self._set_state(opaque_id, "RESOLVED")
        disposition = (
            str(result.get("disposition", "")) if isinstance(result, dict) else ""
        )
        suffix = f" — {disposition}" if disposition else ""
        await self._safe_edit(row, wait, f"{decision}{suffix}")
        return "Hermes gate approved." if action == "a" else "Hermes gate rejected."

    def _set_state(self, opaque_id: str, state: str) -> None:
        self.db.execute(
            "UPDATE prompts SET state=? WHERE opaque_id=?", (state, opaque_id)
        )

    async def _safe_edit(
        self,
        row: sqlite3.Row,
        wait: dict[str, Any],
        status_text: str,
    ) -> None:
        if not row["message_id"]:
            return
        try:
            await self.edit_prompt(
                str(row["chat_id"]), str(row["message_id"]), wait, status_text
            )
        except Exception:
            logger.warning(
                "Failed to update Hermes human-gate Telegram prompt",
                exc_info=True,
            )

    async def _request_json(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None,
    ) -> dict[str, Any]:
        return await asyncio.to_thread(self._request_json_sync, method, path, body)

    def _request_json_sync(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None,
    ) -> dict[str, Any]:
        data = None if body is None else json.dumps(body).encode("utf-8")
        request = urllib.request.Request(
            self.controller_url + path,
            data=data,
            method=method,
            headers={
                "Authorization": f"Bearer {self._operator_token}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            try:
                payload = json.loads(exc.read())
                message = str(payload.get("error") or f"HTTP {exc.code}")
            except Exception:
                message = f"HTTP {exc.code}"
            raise HermesGateHTTPError(exc.code, message) from None
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            raise HermesGateTransportError(str(exc)) from None
        try:
            payload = json.loads(raw)
        except Exception as exc:
            raise HermesGateTransportError(
                "invalid JSON from Hermes Controller"
            ) from exc
        if not isinstance(payload, dict):
            raise HermesGateTransportError("invalid object from Hermes Controller")
        return payload
