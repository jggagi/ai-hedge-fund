from fastapi import APIRouter, HTTPException, Request, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from sqlalchemy import text
import asyncio
from datetime import datetime, timezone
import json
import logging

from app.backend.database import get_db, SessionLocal
from app.backend.database.models import HedgeFundFlowRun
from app.backend.repositories.flow_repository import FlowRepository
from app.backend.repositories.flow_run_repository import FlowRunRepository
from app.backend.models.schemas import ErrorResponse, HedgeFundRequest, BacktestRequest, BacktestDayResult, BacktestPerformanceMetrics, FlowRunStatus
from app.backend.models.events import StartEvent, ProgressUpdateEvent, ErrorEvent, CompleteEvent
from app.backend.services.graph import create_graph, parse_hedge_fund_response, run_graph
from app.backend.services.portfolio import create_portfolio
from app.backend.services.backtest_service import BacktestService
from app.backend.services.api_key_service import ApiKeyService
from app.backend.services.run_manager import research_run_manager
from src.graph.state import AgentState
from src.run_context import RunExecutionContext, _sanitize, use_run_context
from langchain_core.messages import HumanMessage
from src.utils.progress import progress
from src.utils.analysts import get_agents_list

router = APIRouter(prefix="/hedge-fund")
logger = logging.getLogger(__name__)


def _initial_run_state(request_data: HedgeFundRequest, portfolio: dict, context: RunExecutionContext) -> AgentState:
    model_provider = request_data.model_provider.value if hasattr(request_data.model_provider, "value") else request_data.model_provider
    return {
        "messages": [HumanMessage(content="Make trading decisions based on the provided data.")],
        "data": {
            "tickers": request_data.tickers,
            "portfolio": portfolio,
            "start_date": request_data.start_date or request_data.get_start_date(),
            "end_date": request_data.end_date,
            "analyst_signals": {},
        },
        "metadata": {
            "show_reasoning": False,
            "model_name": request_data.model_name,
            "model_provider": model_provider,
            "request": request_data,
            "run_context": context,
            "strict_execution": True,
        },
    }


def _valid_result(result: dict) -> dict:
    if not isinstance(result, dict) or not result.get("messages"):
        raise RuntimeError("Research graph returned no final message")
    final_message = result["messages"][-1]
    content = getattr(final_message, "content", None)
    decisions = parse_hedge_fund_response(content)
    if not isinstance(decisions, dict) or not decisions:
        raise RuntimeError("Research graph returned empty or invalid decisions")
    return {
        "decisions": decisions,
        "analyst_signals": result.get("data", {}).get("analyst_signals", {}),
        "current_prices": result.get("data", {}).get("current_prices", {}),
    }

@router.post(
    path="/run",
    responses={
        200: {"description": "Successful response with streaming updates"},
        400: {"model": ErrorResponse, "description": "Invalid request parameters"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def run(request_data: HedgeFundRequest, request: Request, db: Session = Depends(get_db)):
    if not research_run_manager.reserve_slot():
        raise HTTPException(status_code=409, detail="A one-time research run is already active")

    flow_run = None
    secret_values: tuple[str, ...] = ()
    try:
        if request_data.data_source == "financial_datasets" and not request_data.api_keys:
            request_data.api_keys = ApiKeyService(db).get_api_keys_dict()
        secret_values = tuple((request_data.api_keys or {}).values())
        request_safe = _sanitize(request_data.model_dump(mode="json", exclude={"api_keys"}), secret_values)

        flow_repo = FlowRepository(db)
        if request_data.flow_id is None:
            synthetic_flow = flow_repo.create_flow(
                name=f"Research {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
                description="Automatically created for a one-time research request.",
                nodes=_sanitize([node.model_dump(mode="json", exclude_none=True) for node in request_data.graph_nodes]),
                edges=_sanitize([edge.model_dump(mode="json", exclude_none=True) for edge in request_data.graph_edges]),
                is_template=False,
            )
            request_data.flow_id = synthetic_flow.id
        elif not flow_repo.get_flow_by_id(request_data.flow_id):
            raise HTTPException(status_code=404, detail="Flow not found")
        request_safe["flow_id"] = request_data.flow_id

        db.commit()
        db.execute(text("BEGIN IMMEDIATE"))
        active_count = db.query(HedgeFundFlowRun).filter(
            HedgeFundFlowRun.status.in_([FlowRunStatus.IN_PROGRESS.value, FlowRunStatus.CANCEL_REQUESTED.value])
        ).count()
        if active_count:
            raise HTTPException(status_code=409, detail="A one-time research run is already active")

        flow_run = FlowRunRepository(db).create_flow_run(request_data.flow_id, request_safe)
        run_id = flow_run.id
        FlowRunRepository(db).update_flow_run(run_id, status=FlowRunStatus.IN_PROGRESS)
        run_context = RunExecutionContext(run_id, request_data.flow_id, request_data.timeout_seconds, secret_values)
        portfolio = create_portfolio(
            request_data.initial_cash,
            request_data.margin_requirement,
            request_data.tickers,
            request_data.portfolio_positions,
        )

        if request_data.data_source == "sec_filings":
            from app.backend.services.public_research import create_public_research_graph
            graph = create_public_research_graph(request_data).compile()
        else:
            graph = create_graph(
                graph_nodes=request_data.graph_nodes,
                graph_edges=request_data.graph_edges,
                run_context=run_context,
            ).compile()

        event_queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def publish_progress(agent_name, ticker, status, analysis, timestamp):
            run_context.capture_inference(agent_name, ticker, status, analysis)
            event = ProgressUpdateEvent(
                agent=agent_name,
                ticker=ticker,
                status=status,
                timestamp=timestamp,
                analysis=analysis,
                run_id=run_id,
            )
            loop.call_soon_threadsafe(event_queue.put_nowait, event)

        def work() -> dict:
            progress.register_handler(publish_progress)
            try:
                run_context.check_active()
                with use_run_context(run_context):
                    if request_data.data_source == "sec_filings":
                        result = graph.invoke(_initial_run_state(request_data, portfolio, run_context))
                    else:
                        result = run_graph(
                            graph=graph,
                            portfolio=portfolio,
                            tickers=request_data.tickers,
                            start_date=request_data.start_date or request_data.get_start_date(),
                            end_date=request_data.end_date,
                            model_name=request_data.model_name,
                            model_provider=request_data.model_provider.value,
                            request=request_data,
                            run_context=run_context,
                        )
                run_context.check_active()
                payload = _valid_result(result)
                provenance = run_context.research_report(request_data)
                graph_report = result.get("data", {}).get("research_report")
                if isinstance(graph_report, dict):
                    graph_report["provenance"] = provenance
                    payload["research_report"] = graph_report
                else:
                    payload["research_report"] = provenance
                return payload
            finally:
                progress.unregister_handler(publish_progress)

        def persist_state(status, results, error_message):
            session = SessionLocal()
            try:
                FlowRunRepository(session).update_flow_run(
                    run_id,
                    status=status,
                    results=results if status == FlowRunStatus.COMPLETE else None,
                    error_message=error_message,
                )
            finally:
                session.close()

        active_run = research_run_manager.start(run_context, event_queue, work, persist_state)

        async def wait_for_disconnect():
            try:
                while True:
                    message = await request.receive()
                    if message["type"] == "http.disconnect":
                        return True
            except Exception:
                return True

        async def event_generator():
            disconnect_task = asyncio.create_task(wait_for_disconnect())
            try:
                yield StartEvent(run_id=run_id, flow_id=request_data.flow_id).to_sse()
                # The queue/run task live in the manager, not in this SSE consumer.
                while True:
                    if disconnect_task.done():
                        return
                    try:
                        event = await asyncio.wait_for(event_queue.get(), timeout=0.25)
                    except asyncio.TimeoutError:
                        if active_run.done.is_set() and event_queue.empty():
                            return
                        continue
                    if isinstance(event, ProgressUpdateEvent):
                        yield event.to_sse()
                    elif event.get("kind") == "complete":
                        yield CompleteEvent(data=event["data"], run_id=run_id, flow_id=request_data.flow_id).to_sse()
                        return
                    elif event.get("kind") == "error":
                        yield ErrorEvent(message=event["message"], run_id=run_id, flow_id=request_data.flow_id).to_sse()
                        return
            finally:
                if not disconnect_task.done():
                    disconnect_task.cancel()

        return StreamingResponse(event_generator(), media_type="text/event-stream")
    except HTTPException:
        if flow_run is not None:
            FlowRunRepository(db).update_flow_run(
                flow_run.id,
                status=FlowRunStatus.ERROR,
                error_message="Run setup failed before execution started.",
            )
        research_run_manager.release_reservation()
        raise
    except Exception as e:
        safe_error = str(e)
        for secret in secret_values:
            if secret:
                safe_error = safe_error.replace(secret, "[REDACTED]")
        if flow_run is not None:
            FlowRunRepository(db).update_flow_run(
                flow_run.id,
                status=FlowRunStatus.ERROR,
                error_message=safe_error,
            )
        research_run_manager.release_reservation()
        logger.exception("Failed to set up one-time research run")
        raise HTTPException(status_code=500, detail=f"An error occurred while processing the request: {safe_error}") from e

@router.post(
    path="/backtest",
    responses={
        200: {"description": "Successful response with streaming backtest updates"},
        400: {"model": ErrorResponse, "description": "Invalid request parameters"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def backtest(request_data: BacktestRequest, request: Request, db: Session = Depends(get_db)):
    """Run a continuous backtest over a time period with streaming updates."""
    # One-time research provenance uses a process-wide active run context. Keep
    # backtests mutually exclusive with it so their requests/progress cannot be
    # attributed to another run (or inherit its cancellation/deadline).
    if not research_run_manager.reserve_slot():
        raise HTTPException(status_code=409, detail="Another research or backtest run is already active")
    try:
        # Hydrate API keys from database if not provided
        if not request_data.api_keys:
            api_key_service = ApiKeyService(db)
            request_data.api_keys = api_key_service.get_api_keys_dict()

        # Convert model_provider to string if it's an enum
        model_provider = request_data.model_provider
        if hasattr(model_provider, "value"):
            model_provider = model_provider.value

        # Create the portfolio (same as /run endpoint)
        portfolio = create_portfolio(
            request_data.initial_capital, 
            request_data.margin_requirement, 
            request_data.tickers, 
            request_data.portfolio_positions
        )

        # Construct agent graph using the React Flow graph structure (same as /run endpoint)
        graph = create_graph(graph_nodes=request_data.graph_nodes, graph_edges=request_data.graph_edges)
        graph = graph.compile()

        # Create backtest service with the compiled graph
        backtest_service = BacktestService(
            graph=graph,
            portfolio=portfolio,
            tickers=request_data.tickers,
            start_date=request_data.start_date,
            end_date=request_data.end_date,
            initial_capital=request_data.initial_capital,
            model_name=request_data.model_name,
            model_provider=model_provider,
            request=request_data,  # Pass the full request for agent-specific model access
        )

        # Function to detect client disconnection
        async def wait_for_disconnect():
            """Wait for client disconnect and return True when it happens"""
            try:
                while True:
                    message = await request.receive()
                    if message["type"] == "http.disconnect":
                        return True
            except Exception:
                return True

        # Set up streaming response
        async def event_generator():
            progress_queue = asyncio.Queue()
            backtest_task = None
            disconnect_task = None

            # Global progress handler to capture individual agent updates during backtest
            def progress_handler(agent_name, ticker, status, analysis, timestamp):
                event = ProgressUpdateEvent(agent=agent_name, ticker=ticker, status=status, timestamp=timestamp, analysis=analysis)
                progress_queue.put_nowait(event)

            # Progress callback to handle backtest-specific updates
            def progress_callback(update):
                if update["type"] == "progress":
                    event = ProgressUpdateEvent(
                        agent="backtest",
                        ticker=None,
                        status=f"Processing {update['current_date']} ({update['current_step']}/{update['total_dates']})",
                        timestamp=None,
                        analysis=None
                    )
                    progress_queue.put_nowait(event)
                elif update["type"] == "backtest_result":
                    # Convert day result to a streaming event
                    backtest_result = BacktestDayResult(**update["data"])
                    
                    # Send the full day result data as JSON in the analysis field
                    import json
                    analysis_data = json.dumps(update["data"])
                    
                    event = ProgressUpdateEvent(
                        agent="backtest",
                        ticker=None,
                        status=f"Completed {backtest_result.date} - Portfolio: ${backtest_result.portfolio_value:,.2f}",
                        timestamp=None,
                        analysis=analysis_data
                    )
                    progress_queue.put_nowait(event)

            # Register our handler with the progress tracker to capture agent updates
            progress.register_handler(progress_handler)
            
            try:
                # Start the backtest in a background task
                backtest_task = asyncio.create_task(
                    backtest_service.run_backtest_async(progress_callback=progress_callback)
                )
                
                # Start the disconnect detection task
                disconnect_task = asyncio.create_task(wait_for_disconnect())
                
                # Send initial message
                yield StartEvent().to_sse()

                # Stream progress updates until backtest_task completes or client disconnects
                while not backtest_task.done():
                    # Check if client disconnected
                    if disconnect_task.done():
                        print("Client disconnected, cancelling backtest execution")
                        backtest_task.cancel()
                        try:
                            await backtest_task
                        except asyncio.CancelledError:
                            pass
                        return

                    # Either get a progress update or wait a bit
                    try:
                        event = await asyncio.wait_for(progress_queue.get(), timeout=1.0)
                        yield event.to_sse()
                    except asyncio.TimeoutError:
                        # Just continue the loop
                        pass

                # Get the final result
                try:
                    result = await backtest_task
                except asyncio.CancelledError:
                    print("Backtest task was cancelled")
                    return

                if not result:
                    yield ErrorEvent(message="Failed to complete backtest").to_sse()
                    return

                # Send the final result
                performance_metrics = BacktestPerformanceMetrics(**result["performance_metrics"])
                final_data = CompleteEvent(
                    data={
                        "performance_metrics": performance_metrics.model_dump(),
                        "final_portfolio": result["final_portfolio"],
                        "total_days": len(result["results"]),
                    }
                )
                yield final_data.to_sse()

            except asyncio.CancelledError:
                print("Backtest event generator cancelled")
                return
            finally:
                # Clean up
                progress.unregister_handler(progress_handler)
                if backtest_task and not backtest_task.done():
                    backtest_task.cancel()
                    try:
                        await backtest_task
                    except asyncio.CancelledError:
                        pass
                if disconnect_task and not disconnect_task.done():
                    disconnect_task.cancel()
                research_run_manager.release_reservation()

        # Return a streaming response
        return StreamingResponse(event_generator(), media_type="text/event-stream")

    except HTTPException as e:
        research_run_manager.release_reservation()
        raise e
    except Exception as e:
        research_run_manager.release_reservation()
        raise HTTPException(status_code=500, detail=f"An error occurred while processing the backtest request: {str(e)}")


@router.get(
    path="/agents",
    responses={
        200: {"description": "List of available agents"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def get_agents():
    """Get the list of available agents."""
    try:
        return {"agents": get_agents_list()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve agents: {str(e)}")
