"""Cooperative control and provenance collection for backend one-time runs."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import threading
import time
from typing import Any, Iterator
from urllib.parse import parse_qsl, urlsplit


class RunStopped(Exception):
    """Raised at a safe boundary when a run is cancelled or expires."""


class RunTimedOut(RunStopped):
    """Raised when the configured run deadline has elapsed."""


_active_lock = threading.RLock()
_active_context: "RunExecutionContext | None" = None
_SENSITIVE_PARTS = ("api_key", "apikey", "authorization", "token", "secret", "password", "credential")
_SAFE_QUERY_KEYS = {
    "ticker", "start_date", "end_date", "report_period_lte", "report_period_gte",
    "filing_date_lte", "filing_date_gte", "interval", "interval_multiplier",
    "period", "limit", "line_items",
}


def _sanitize(value: Any, secret_values: tuple[str, ...] = ()) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _sanitize(item, secret_values)
            for key, item in value.items()
            if not any(part in str(key).lower() for part in _SENSITIVE_PARTS)
        }
    if isinstance(value, (list, tuple)):
        return [_sanitize(item, secret_values) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, str):
            for secret in secret_values:
                if secret:
                    value = value.replace(secret, "[REDACTED]")
        return value
    if hasattr(value, "model_dump"):
        return _sanitize(value.model_dump(mode="json"), secret_values)
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


class RunExecutionContext:
    def __init__(self, run_id: int, flow_id: int, timeout_seconds: int, secret_values: tuple[str, ...] = ()):
        self.run_id = run_id
        self.flow_id = flow_id
        self.timeout_seconds = timeout_seconds
        self._secret_values = tuple(secret for secret in secret_values if secret)
        self.deadline = time.monotonic() + timeout_seconds
        self.cancel_event = threading.Event()
        self._stop_lock = threading.Lock()
        self._stop_reason: str | None = None
        self._data_lock = threading.Lock()
        self.source_snapshots: list[dict[str, Any]] = []
        self.model_invocations: list[dict[str, Any]] = []
        self.agent_inferences: list[dict[str, Any]] = []
        self.data_gaps: list[dict[str, Any]] = []
        self.omitted_source_snapshots = 0
        self._model_digest_cache: dict[str, str] = {}

    def get_model_digest(self, model_name: str, lookup: Any) -> str:
        """Read a local model's immutable digest at most once per run."""
        with self._data_lock:
            if model_name in self._model_digest_cache:
                return self._model_digest_cache[model_name]
            try:
                digest = lookup(model_name) or "unknown"
            except Exception:
                digest = "unknown"
            self._model_digest_cache[model_name] = str(digest)
            return self._model_digest_cache[model_name]

    @property
    def stop_reason(self) -> str | None:
        with self._stop_lock:
            return self._stop_reason

    def request_stop(self, reason: str) -> None:
        if reason not in {"cancel", "timeout"}:
            raise ValueError("reason must be cancel or timeout")
        with self._stop_lock:
            if self._stop_reason is None:
                self._stop_reason = reason
            self.cancel_event.set()

    def remaining_seconds(self) -> float:
        return max(0.1, self.deadline - time.monotonic())

    def redact_text(self, value: str) -> str:
        for secret in self._secret_values:
            value = value.replace(secret, "[REDACTED]")
        return value

    def check_active(self) -> None:
        if self.stop_reason == "timeout" or time.monotonic() >= self.deadline:
            self.request_stop("timeout")
            raise RunTimedOut("Research run exceeded its configured deadline")
        if self.cancel_event.is_set():
            raise RunStopped("Research run was cancelled")

    def interruptible_sleep(self, seconds: float) -> None:
        if self.cancel_event.wait(min(seconds, self.remaining_seconds())):
            self.check_active()
        self.check_active()

    def capture_source(
        self,
        method: str,
        url: str,
        *,
        status_code: int | None = None,
        body: Any = None,
        request_data: Any = None,
        error: str | None = None,
        cache_hit: bool = False,
    ) -> None:
        parts = urlsplit(url)
        safe_query = {
            key: value
            for key, value in parse_qsl(parts.query, keep_blank_values=True)
            if key.lower() in _SAFE_QUERY_KEYS
        }
        snapshot: dict[str, Any] = {
            "source": parts.netloc,
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "method": method.upper(),
            "resource_location": f"{parts.scheme}://{parts.netloc}{parts.path}",
            "parameters": safe_query,
            "cache_hit": cache_hit,
            "record_type": "agent_tool_result" if method.upper() == "TOOL_RESULT" else "http_response",
        }
        if status_code is not None:
            snapshot["status_code"] = status_code
            if status_code >= 400:
                with self._data_lock:
                    self.data_gaps.append({
                        "resource_location": snapshot["resource_location"],
                        "reason": f"source returned HTTP {status_code}",
                    })
        if error:
            snapshot["error"] = error
            gap = {"resource_location": snapshot["resource_location"], "error": error}
            with self._data_lock:
                self.data_gaps.append(gap)
        if request_data is not None:
            snapshot["request_parameters"] = _sanitize(request_data, self._secret_values)
        if body is not None:
            safe_body = _sanitize(body, self._secret_values)
            encoded = json.dumps(safe_body, ensure_ascii=False, default=str)
            snapshot["facts_sha256"] = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
            snapshot["facts_size_bytes"] = len(encoded.encode("utf-8"))
            if len(encoded) > 32_000:
                snapshot["facts"] = encoded[:32_000]
                snapshot["facts_truncated"] = True
            else:
                snapshot["facts"] = safe_body
            if not safe_body:
                with self._data_lock:
                    self.data_gaps.append({"resource_location": snapshot["resource_location"], "reason": "empty response"})
        with self._data_lock:
            if len(self.source_snapshots) < 200:
                self.source_snapshots.append(snapshot)
            else:
                self.omitted_source_snapshots += 1

    def capture_model(
        self,
        model_name: str,
        model_provider: Any,
        prompt: Any,
        *,
        schema: dict[str, Any],
        temperature: float | None,
        options: dict[str, Any],
        model_digest: str | None = None,
    ) -> None:
        prompt_text = prompt.to_string() if hasattr(prompt, "to_string") else str(prompt)
        provider = str(getattr(model_provider, "value", model_provider))
        prompt_hash = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()
        schema_json = json.dumps(schema, sort_keys=True, ensure_ascii=False, default=str)
        schema_hash = hashlib.sha256(schema_json.encode("utf-8")).hexdigest()
        safe_options = _sanitize(options)
        config = {
            "model_name": model_name,
            "model_provider": provider,
            "temperature": temperature,
            "options": safe_options,
            "model_digest": model_digest,
            "schema_sha256": schema_hash,
            "prompt_sha256": prompt_hash,
        }
        entry = {
            **config,
            "model_config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest(),
            "prompt_sha256": prompt_hash,
            "schema_sha256": schema_hash,
            "prompt_size_bytes": len(prompt_text.encode("utf-8")),
            "temperature_mode": "provider_default" if temperature is None else "explicit",
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }
        with self._data_lock:
            self.model_invocations.append(entry)

    def capture_inference(self, agent: str, ticker: str | None, status: str, analysis: str | None) -> None:
        if not analysis:
            return
        with self._data_lock:
            self.agent_inferences.append({
                "agent": agent,
                "ticker": ticker,
                "status": status,
                "recorded_at": datetime.now(timezone.utc).isoformat(),
                "inference": self.redact_text(analysis),
            })

    def research_report(self, request: Any) -> dict[str, Any]:
        start_date = request.start_date or request.get_start_date()
        model_config = {
            "global": {
                "model_name": request.model_name,
                "model_provider": request.model_provider.value if hasattr(request.model_provider, "value") else request.model_provider,
            },
            "agents": [
                {
                    "agent_id": node.id,
                    "model_name": request.get_agent_model_config(node.id)[0],
                    "model_provider": request.get_agent_model_config(node.id)[1].value
                    if hasattr(request.get_agent_model_config(node.id)[1], "value")
                    else request.get_agent_model_config(node.id)[1],
                }
                for node in request.graph_nodes
            ],
        }
        with self._data_lock:
            snapshots = list(self.source_snapshots)
            invocations = list(self.model_invocations)
            inferences = list(self.agent_inferences)
            gaps = list(self.data_gaps)
        return {
            "run_id": self.run_id,
            "flow_id": self.flow_id,
            "data_window": {"tickers": list(request.tickers), "start_date": start_date, "end_date": request.end_date},
            "data_source": request.data_source,
            "model_config": model_config,
            "source_snapshots": snapshots,
            "source_snapshots_omitted": self.omitted_source_snapshots,
            "model_invocations": invocations,
            "agent_inferences": inferences,
            "data_gaps": gaps,
        }


@contextmanager
def use_run_context(context: RunExecutionContext) -> Iterator[None]:
    global _active_context
    with _active_lock:
        previous = _active_context
        _active_context = context
    try:
        yield
    finally:
        with _active_lock:
            if _active_context is context:
                _active_context = previous


def get_active_run_context() -> RunExecutionContext | None:
    with _active_lock:
        return _active_context
