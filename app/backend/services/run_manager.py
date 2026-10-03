"""In-process coordination for persisted one-time research runs."""

from __future__ import annotations

import asyncio
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
import threading
import time
import logging
from typing import Any, Callable

from app.backend.models.schemas import FlowRunStatus
from src.run_context import RunExecutionContext


@dataclass
class ResearchRun:
    context: RunExecutionContext
    loop: asyncio.AbstractEventLoop
    event_queue: asyncio.Queue
    done: asyncio.Event = field(default_factory=asyncio.Event)
    worker: Future | None = None
    task: asyncio.Task | None = None
    final_status: FlowRunStatus | None = None
    final_data: dict[str, Any] | None = None
    final_error: str | None = None

    def publish(self, event: Any) -> None:
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None
        if current_loop is self.loop:
            self.event_queue.put_nowait(event)
        else:
            self.loop.call_soon_threadsafe(self.event_queue.put_nowait, event)


class ResearchRunManager:
    """Keep graph work alive independently from its SSE consumer."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._reserved = False
        self._runs: dict[int, ResearchRun] = {}
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hedge-research")

    def reserve_slot(self) -> bool:
        with self._lock:
            if self._reserved:
                return False
            self._reserved = True
            return True

    def release_reservation(self) -> None:
        with self._lock:
            if not self._runs:
                self._reserved = False

    def start(
        self,
        context: RunExecutionContext,
        event_queue: asyncio.Queue,
        work: Callable[[], dict[str, Any]],
        persist_terminal: Callable[[FlowRunStatus, dict[str, Any] | None, str | None], None],
    ) -> ResearchRun:
        loop = asyncio.get_running_loop()
        run = ResearchRun(context=context, loop=loop, event_queue=event_queue)
        with self._lock:
            self._runs[context.run_id] = run
            self._reserved = True
        run.worker = self._executor.submit(work)
        run.task = asyncio.create_task(self._watch(run, persist_terminal))
        return run

    async def _watch(
        self,
        run: ResearchRun,
        persist_terminal: Callable[[FlowRunStatus, dict[str, Any] | None, str | None], None],
    ) -> None:
        assert run.worker is not None
        while not run.worker.done():
            if run.context.stop_reason is None and time.monotonic() >= run.context.deadline:
                run.context.request_stop("timeout")
                try:
                    persist_terminal(
                        FlowRunStatus.CANCEL_REQUESTED,
                        None,
                        "Run deadline elapsed; waiting for in-flight work to stop.",
                    )
                except Exception:
                    logger.exception("Could not persist timeout request for research run %s", run.context.run_id)
            await asyncio.sleep(0.05)

        try:
            result = run.worker.result()
            if run.context.stop_reason == "timeout":
                status, data, error = FlowRunStatus.TIMED_OUT, None, "Research run exceeded its configured deadline."
            elif run.context.stop_reason == "cancel":
                status, data, error = FlowRunStatus.CANCELLED, None, "Research run was cancelled."
            else:
                status, data, error = FlowRunStatus.COMPLETE, result, None
        except BaseException as exc:
            if run.context.stop_reason == "timeout" or exc.__class__.__name__ == "RunTimedOut":
                status, data, error = FlowRunStatus.TIMED_OUT, None, "Research run exceeded its configured deadline."
            elif run.context.stop_reason == "cancel" or exc.__class__.__name__ == "RunStopped":
                status, data, error = FlowRunStatus.CANCELLED, None, "Research run was cancelled."
            else:
                detail = run.context.redact_text(str(exc))
                status, data, error = FlowRunStatus.ERROR, None, f"{exc.__class__.__name__}: {detail}"[:2000]

        try:
            persist_terminal(status, data, error)
        except Exception as exc:
            logger.exception("Could not persist terminal state for research run %s", run.context.run_id)
            status = FlowRunStatus.ERROR
            data = None
            error = f"Could not persist research result: {run.context.redact_text(str(exc))}"[:2000]
            try:
                persist_terminal(status, None, error)
            except Exception:
                logger.exception("Could not persist fallback error for research run %s", run.context.run_id)
        run.final_status = status
        run.final_data = data
        run.final_error = error
        if status == FlowRunStatus.COMPLETE:
            run.event_queue.put_nowait({"kind": "complete", "data": data})
        else:
            run.event_queue.put_nowait({"kind": "error", "message": error or status.value})
        run.done.set()
        with self._lock:
            self._runs.pop(run.context.run_id, None)
            self._reserved = bool(self._runs)

    def request_cancel(self, flow_id: int, run_id: int) -> ResearchRun | None:
        with self._lock:
            run = self._runs.get(run_id)
            if not run or run.context.flow_id != flow_id:
                return None
            if run.worker is not None and run.worker.done():
                return run
            run.context.request_stop("cancel")
            return run

    def is_active(self) -> bool:
        with self._lock:
            return self._reserved

    def owns_run(self, run_id: int) -> bool:
        with self._lock:
            return run_id in self._runs

    def owns_flow(self, flow_id: int) -> bool:
        with self._lock:
            return any(run.context.flow_id == flow_id for run in self._runs.values())


research_run_manager = ResearchRunManager()
logger = logging.getLogger(__name__)
