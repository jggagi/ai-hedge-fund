import asyncio
import json
import threading
import time

from langchain_core.messages import HumanMessage
import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from starlette.requests import Request

from app.backend.database.models import Base, HedgeFundFlowRun
from app.backend.models.schemas import FlowRunStatus, HedgeFundRequest
from app.backend.models.schemas import FlowRunUpdateRequest
from app.backend.repositories.flow_repository import FlowRepository
from app.backend.repositories.flow_run_repository import FlowRunRepository
from app.backend.routes import flow_runs, hedge_fund
from app.backend.services.run_manager import ResearchRunManager
from src.run_context import RunExecutionContext
from src.tools import api as financial_api
from app.backend.services import graph as graph_service


_TEST_ENGINE = create_engine("sqlite://", connect_args={"check_same_thread": False})
_TestSession = sessionmaker(autocommit=False, autoflush=False, bind=_TEST_ENGINE)
Base.metadata.create_all(bind=_TEST_ENGINE)
hedge_fund.SessionLocal = _TestSession

def _request(tickers=None, **overrides):
    payload = {
        "tickers": tickers or ["ZZZ_LIFECYCLE"],
        "graph_nodes": [{"id": "fundamentals_abc123", "type": "agent", "data": {}}],
        "graph_edges": [],
        "start_date": "2024-01-01",
        "end_date": "2024-12-31",
        "api_keys": {"FINANCIAL_DATASETS_API_KEY": "LIFECYCLE_TEST_SECRET"},
    }
    payload.update(overrides)
    return HedgeFundRequest(**payload)


def _direct_request(receive):
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/hedge-fund/run",
        "raw_path": b"/hedge-fund/run",
        "query_string": b"",
        "headers": [],
        "client": ("testclient", 12345),
        "server": ("testserver", 80),
        "root_path": "",
    }
    return Request(scope, receive)


def _new_session():
    Base.metadata.drop_all(bind=_TEST_ENGINE)
    Base.metadata.create_all(bind=_TEST_ENGINE)
    return _TestSession()


def test_one_time_run_persists_sse_provenance_without_api_keys(monkeypatch):
    class FakeResponse:
        status_code = 200

        @staticmethod
        def json():
            return {"ticker": "ZZZ_LIFECYCLE", "prices": []}

    monkeypatch.setattr(financial_api.requests, "get", lambda *args, **kwargs: FakeResponse())

    class FakeGraph:
        def compile(self):
            return self

        def invoke(self, _state):
            financial_api.get_prices("ZZZ_LIFECYCLE", "2024-01-01", "2024-12-31", "LIFECYCLE_TEST_SECRET")
            return {
                "messages": [HumanMessage(content=json.dumps({"ZZZ_LIFECYCLE": {"action": "hold"}}))],
                "data": {"analyst_signals": {"mock": {"signal": "neutral"}}, "current_prices": {}},
            }

    monkeypatch.setattr(hedge_fund, "create_graph", lambda **_kwargs: FakeGraph())
    db = _new_session()

    async def never_disconnect():
        await asyncio.sleep(60)
        return {"type": "http.disconnect"}

    async def scenario():
        response = await hedge_fund.run(_request(), _direct_request(never_disconnect), db)
        events = []
        async for event in response.body_iterator:
            events.append(event)
        return events

    try:
        events = asyncio.run(scenario())
        combined = "".join(events)
        start_data = json.loads(combined.split("event: start\ndata: ", 1)[1].split("\n\n", 1)[0])
        complete_data = json.loads(combined.split("event: complete\ndata: ", 1)[1].split("\n\n", 1)[0])
        run_id = start_data["run_id"]
        assert start_data["flow_id"] is not None
        assert complete_data["run_id"] == run_id

        run = FlowRunRepository(db).get_flow_run_by_id(run_id)
        assert run.status == FlowRunStatus.COMPLETE.value
        assert "api_keys" not in run.request_data
        assert "LIFECYCLE_TEST_SECRET" not in json.dumps(run.request_data)
        report = run.results["research_report"]
        raw_source = next(item for item in report["source_snapshots"] if item["record_type"] == "http_response")
        typed_source = next(item for item in report["source_snapshots"] if item["record_type"] == "agent_tool_result")
        assert raw_source["source"] == "api.financialdatasets.ai"
        assert raw_source["status_code"] == 200
        assert raw_source["facts_sha256"]
        assert typed_source["facts"] == []
        assert report["data_window"]["tickers"] == ["ZZZ_LIFECYCLE"]
        assert "LIFECYCLE_TEST_SECRET" not in json.dumps(run.results)
    finally:
        db.close()


def test_stream_disconnect_does_not_cancel_persisted_run(monkeypatch):
    worker_done = threading.Event()

    class FakeGraph:
        def compile(self):
            return self

        def invoke(self, _state):
            time.sleep(0.1)
            worker_done.set()
            return {"messages": [HumanMessage(content='{"AAPL":{"action":"hold"}}')], "data": {}}

    monkeypatch.setattr(hedge_fund, "create_graph", lambda **_kwargs: FakeGraph())
    db = _new_session()

    async def disconnect():
        return {"type": "http.disconnect"}

    async def scenario():
        response = await hedge_fund.run(_request(["AAPL"]), _direct_request(disconnect), db)
        iterator = response.body_iterator
        first = await anext(iterator)
        assert "event: start" in first
        try:
            await anext(iterator)
        except StopAsyncIteration:
            pass
        deadline = time.monotonic() + 3
        while not worker_done.is_set() and time.monotonic() < deadline:
            await asyncio.sleep(0.02)
        run_id = int(json.loads(first.split("data: ", 1)[1])["run_id"])
        while time.monotonic() < deadline:
            db.expire_all()
            run = FlowRunRepository(db).get_flow_run_by_id(run_id)
            if run.status == FlowRunStatus.COMPLETE.value:
                break
            await asyncio.sleep(0.02)
        return run

    try:
        run = asyncio.run(scenario())
        assert worker_done.is_set()
        assert run.status == FlowRunStatus.COMPLETE.value
        assert run.results["decisions"]["AAPL"]["action"] == "hold"
    finally:
        db.close()


def test_cancel_endpoint_waits_for_worker_before_terminal_state(monkeypatch):
    worker_entered = threading.Event()
    worker_exited = threading.Event()

    class FakeGraph:
        def compile(self):
            return self

        def invoke(self, state):
            context = state["metadata"]["run_context"]
            worker_entered.set()
            context.cancel_event.wait(timeout=3)
            time.sleep(0.15)
            worker_exited.set()
            context.check_active()

    monkeypatch.setattr(hedge_fund, "create_graph", lambda **_kwargs: FakeGraph())
    db = _new_session()
    flow = FlowRepository(db).create_flow("cancel fixture", [], [])

    async def wait_for_request():
        await asyncio.sleep(60)
        return {"type": "http.disconnect"}

    async def scenario():
        response = await hedge_fund.run(_request(["AAPL"], flow_id=flow.id), _direct_request(wait_for_request), db)
        iterator = response.body_iterator
        first = await anext(iterator)
        event = json.loads(first.split("data: ", 1)[1])
        run_id = event["run_id"]
        deadline = time.monotonic() + 2
        while not worker_entered.is_set() and time.monotonic() < deadline:
            await asyncio.sleep(0.01)
        requested = await flow_runs.cancel_flow_run(flow.id, run_id, db)
        assert requested.status == FlowRunStatus.CANCEL_REQUESTED
        assert not worker_exited.is_set()
        db.expire_all()
        intermediate = FlowRunRepository(db).get_flow_run_by_id(run_id)
        assert intermediate.status == FlowRunStatus.CANCEL_REQUESTED.value
        terminal_events = []
        async for item in iterator:
            terminal_events.append(item)
        db.expire_all()
        final = FlowRunRepository(db).get_flow_run_by_id(run_id)
        return final, terminal_events

    try:
        final, terminal_events = asyncio.run(scenario())
        assert worker_exited.is_set()
        assert final.status == FlowRunStatus.CANCELLED.value
        assert final.completed_at is not None
        assert any("event: error" in item and '"run_id"' in item for item in terminal_events)
    finally:
        db.close()


def test_timeout_waits_for_cooperative_worker_to_exit():
    async def scenario():
        manager = ResearchRunManager()
        assert manager.reserve_slot()
        context = RunExecutionContext(991001, 991, 0.05)
        worker_exited = threading.Event()
        states = []

        def work():
            context.cancel_event.wait(timeout=1)
            time.sleep(0.1)
            worker_exited.set()
            context.check_active()

        run = manager.start(context, asyncio.Queue(), work, lambda status, _results, _error: states.append(status))
        await asyncio.wait_for(run.done.wait(), timeout=2)
        assert worker_exited.is_set()
        assert FlowRunStatus.CANCEL_REQUESTED in states
        assert run.final_status == FlowRunStatus.TIMED_OUT

    asyncio.run(scenario())


def test_backtest_and_one_time_run_share_the_same_exclusive_slot():
    manager = hedge_fund.research_run_manager
    assert manager.reserve_slot()
    try:
        with pytest.raises(HTTPException) as error:
            asyncio.run(hedge_fund.backtest(None, None, None))
        assert error.value.status_code == 409
    finally:
        manager.release_reservation()


def test_active_run_cannot_be_updated_or_deleted_through_legacy_crud():
    db = _new_session()
    try:
        flow = FlowRepository(db).create_flow("protected active run", [], [])
        run_repo = FlowRunRepository(db)
        run = run_repo.create_flow_run(flow.id, {})
        run_repo.update_flow_run(run.id, status=FlowRunStatus.IN_PROGRESS)

        async def assert_conflicts():
            with pytest.raises(HTTPException) as update_error:
                await flow_runs.update_flow_run(flow.id, run.id, FlowRunUpdateRequest(status=FlowRunStatus.COMPLETE), db)
            with pytest.raises(HTTPException) as delete_error:
                await flow_runs.delete_flow_run(flow.id, run.id, db)
            with pytest.raises(HTTPException) as delete_all_error:
                await flow_runs.delete_all_flow_runs(flow.id, db)
            with pytest.raises(HTTPException) as delete_flow_error:
                from app.backend.routes import flows
                await flows.delete_flow(flow.id, db)
            assert [update_error.value.status_code, delete_error.value.status_code,
                    delete_all_error.value.status_code, delete_flow_error.value.status_code] == [409] * 4

        asyncio.run(assert_conflicts())
        assert run_repo.get_flow_run_by_id(run.id).status == FlowRunStatus.IN_PROGRESS.value
    finally:
        db.close()


def test_cancelling_async_graph_waits_for_executor_thread(monkeypatch):
    import threading

    entered = threading.Event()
    finish = threading.Event()
    exited = threading.Event()

    def slow_graph(*_args):
        entered.set()
        finish.wait(timeout=2)
        exited.set()
        return {"ok": True}

    monkeypatch.setattr(graph_service, "run_graph", slow_graph)

    async def scenario():
        task = asyncio.create_task(graph_service.run_graph_async(None, {}, [], "", "", "", ""))
        while not entered.is_set():
            await asyncio.sleep(0.005)
        task.cancel()
        await asyncio.sleep(0.03)
        assert not exited.is_set()
        assert not task.done()
        finish.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert exited.is_set()

    asyncio.run(scenario())


def test_restart_marks_in_progress_and_cancel_requested_runs_interrupted():
    db = _new_session()
    try:
        flow = FlowRepository(db).create_flow("restart fixture", [], [])
        repository = FlowRunRepository(db)
        in_progress = repository.create_flow_run(flow.id, {})
        repository.update_flow_run(in_progress.id, status=FlowRunStatus.IN_PROGRESS)
        cancel_pending = repository.create_flow_run(flow.id, {})
        repository.update_flow_run(cancel_pending.id, status=FlowRunStatus.CANCEL_REQUESTED)

        assert repository.mark_interrupted_runs() == 2
        assert repository.get_flow_run_by_id(in_progress.id).status == FlowRunStatus.ERROR.value
        assert repository.get_flow_run_by_id(cancel_pending.id).status == FlowRunStatus.ERROR.value
        assert repository.get_flow_run_by_id(in_progress.id).completed_at is not None
    finally:
        db.close()


def test_model_digest_lookup_is_cached_and_unknown_is_explicit():
    context = RunExecutionContext(991002, 991, 30)
    calls = []

    def lookup(model_name):
        calls.append(model_name)
        return None

    first = context.get_model_digest("qwen3.5:0.8b", lookup)
    second = context.get_model_digest("qwen3.5:0.8b", lookup)
    assert first == second == "unknown"
    assert calls == ["qwen3.5:0.8b"]

    context.capture_model(
        "qwen3.5:0.8b", "Ollama", "prompt", schema={}, temperature=0,
        options={"num_predict": 768}, model_digest=first,
    )
    assert context.model_invocations[0]["model_digest"] == "unknown"


def test_strict_ollama_records_mocked_tags_digest_once_per_run(monkeypatch):
    from pydantic import BaseModel
    from src.utils import llm as llm_utils

    class Output(BaseModel):
        value: str

    class FakeModel:
        def with_structured_output(self, *_args, **_kwargs):
            return self

        def invoke(self, _prompt):
            return Output(value="mocked")

    lookups = []
    monkeypatch.setattr(llm_utils, "_lookup_ollama_model_digest", lambda name: lookups.append(name) or "sha256:mocked")
    monkeypatch.setattr(llm_utils, "get_model", lambda *args, **kwargs: FakeModel())
    monkeypatch.setattr(llm_utils, "get_model_info", lambda *args, **kwargs: None)
    context = RunExecutionContext(991003, 991, 30)
    state = {"metadata": {
        "strict_execution": True,
        "run_context": context,
        "model_provider": "Ollama",
        "model_name": "qwen3.5:0.8b",
        "request": {"api_keys": {}},
    }}

    llm_utils.call_llm("mock prompt", Output, agent_name="mock", state=state)
    llm_utils.call_llm("second prompt", Output, agent_name="mock", state=state)

    assert lookups == ["qwen3.5:0.8b"]
    assert [entry["model_digest"] for entry in context.model_invocations] == ["sha256:mocked"] * 2


def test_ollama_digest_lookup_uses_configured_model_base_url(monkeypatch):
    import ollama
    from types import SimpleNamespace
    from src.utils.llm import _lookup_ollama_model_digest

    constructed_hosts = []

    class FakeClient:
        def __init__(self, *, host, timeout):
            constructed_hosts.append((host, timeout))

        def list(self):
            return SimpleNamespace(models=[SimpleNamespace(model="qwen3.5:0.8b", digest="sha256:mocked")])

    monkeypatch.setattr(ollama, "Client", FakeClient)
    monkeypatch.setenv("OLLAMA_HOST", "ollama.internal")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://custom-ollama:11435")
    assert _lookup_ollama_model_digest("qwen3.5:0.8b") == "sha256:mocked"
    monkeypatch.delenv("OLLAMA_BASE_URL")
    assert _lookup_ollama_model_digest("qwen3.5:0.8b") == "sha256:mocked"

    assert constructed_hosts == [
        ("http://custom-ollama:11435", 1.0),
        ("http://ollama.internal:11434", 1.0),
    ]


def test_strict_run_llm_failure_never_returns_default(monkeypatch):
    from src.utils import llm as llm_utils

    class FakeModel:
        def with_structured_output(self, *_args, **_kwargs):
            return self

        def invoke(self, _prompt):
            raise TimeoutError("mock provider timeout")

    monkeypatch.setattr(llm_utils, "get_model", lambda *args, **kwargs: FakeModel())
    monkeypatch.setattr(llm_utils, "get_model_info", lambda *args, **kwargs: None)

    class Output:
        model_fields = {}

        @classmethod
        def model_json_schema(cls):
            return {"type": "object"}

    context = RunExecutionContext(991002, 991, 30)
    state = {"metadata": {"strict_execution": True, "run_context": context, "model_provider": "OpenAI"}}
    try:
        llm_utils.call_llm("mock prompt", Output, agent_name="mock", state=state, max_retries=1)
    except RuntimeError as exc:
        assert "LLM analysis failed" in str(exc)
    else:
        raise AssertionError("strict run returned a default after provider failure")
