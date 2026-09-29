"""CampusMind FastAPI 入口。

当前版本：HTTP /chat + SSE /chat/stream -> Orchestrator -> Agent Runtime -> Guard -> Response，
并支持 Redis 状态/Trace 后端、Chroma Vector RAG 与极薄 Vue Demo。

说明：/chat/stream 提供真实 SSE 事件流和增量 UI 渲染，但当前 LLMClient 仍使用非流式
Chat Completions。也就是说，Agent 完整结果产生后再分块发送 response；不要把它描述成
“模型 token-level streaming”。这样保留 Runtime / Tool Calling 的现有确定性边界，同时让
前端具备 Agent 事件流展示能力。
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse

from agents.agent_orchestrator import AgentOrchestrator, Router
from agents.factory import build_agents
from agents.tools import build_default_tool_registry
from core.config import Settings, build_llm_client
from core.intent_analyzer import IntentAnalyzer
from core.llm_client import BaseLLMClient
from core.risk_analyzer import RiskAnalyzer
from core.risk_policy import RiskPolicyEngine
from core.schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    ToolTraceItem,
    TraceRecord,
)
from observability.factory import build_trace_store
from observability.trace_store import TraceStore
from rag.factory import build_resource_store
from rag.resource_store import ResourceStore
from safety.response_guard import ResponseGuard
from state.case_store import CaseStore
from state.factory import build_state_stores
from state.memory_store import ConversationMemory

APP_VERSION = "1.2.0"
WEB_DIR = Path(__file__).resolve().parent / "web"


class UTF8JSONResponse(JSONResponse):
    """为 Windows PowerShell 5.1 等旧客户端显式声明 UTF-8。"""

    media_type = "application/json; charset=utf-8"


def build_orchestrator(
    settings: Settings | None = None,
    *,
    llm_client: BaseLLMClient | None = None,
    memory_store: ConversationMemory | None = None,
    case_store: CaseStore | None = None,
    resource_store: ResourceStore | None = None,
) -> AgentOrchestrator:
    """构建一套独立的 CampusMind Orchestrator。

    Settings() 默认 static + in_memory + keyword，确保单元测试完全离线。
    FastAPI 正式启动时使用 Settings.from_env()；Docker Compose 默认覆盖为 Redis + Chroma。
    """
    settings = settings or Settings()

    if memory_store is None or case_store is None:
        default_memory, default_cases = build_state_stores(settings)
        memory_store = memory_store or default_memory
        case_store = case_store or default_cases
    resource_store = resource_store or build_resource_store(settings)

    registry = build_default_tool_registry(
        case_store=case_store,
        resource_store=resource_store,
    )
    llm_client = llm_client or build_llm_client(settings)
    agents = build_agents(llm_client=llm_client, tool_registry=registry)

    return AgentOrchestrator(
        intent_analyzer=IntentAnalyzer(),
        risk_analyzer=RiskAnalyzer(),
        policy_engine=RiskPolicyEngine(),
        router=Router(agents),
        memory_store=memory_store,
        case_store=case_store,
        response_guard=ResponseGuard(),
    )


def _tool_trace_items(result) -> list[ToolTraceItem]:
    return [
        ToolTraceItem(
            name=item.name,
            source=item.source,
            status=item.status,
            argument_keys=sorted(item.arguments.keys()),
            error=item.error,
        )
        for item in result.agent_result.tool_executions
    ]


def _trace_record(request_id: str, result, latency_ms: float) -> TraceRecord:
    guard = result.agent_result.guard_result
    return TraceRecord(
        request_id=request_id,
        status="success",
        intent=result.intent_result.intent,
        risk=result.risk_assessment.risk_level,
        allowed_agent=result.policy_decision.allowed_agent,
        selected_agent=result.selected_agent,
        tool_executions=_tool_trace_items(result),
        guard_passed=(guard.passed if guard is not None else None),
        guard_violations=(guard.violations if guard is not None else []),
        safe_fallback_used=result.agent_result.safe_fallback_used,
        memory_size=result.memory_size,
        case_status=result.case_state.status,
        latency_ms=latency_ms,
    )


def _chat_response(request_id: str, result) -> ChatResponse:
    guard = result.agent_result.guard_result
    return ChatResponse(
        request_id=request_id,
        response=result.agent_result.response,
        intent=result.intent_result.intent,
        risk=result.risk_assessment.risk_level,
        selected_agent=result.selected_agent,
        tools_used=result.agent_result.tools_used,
        guard_passed=(guard.passed if guard is not None else None),
        guard_violations=(guard.violations if guard is not None else []),
        safe_fallback_used=result.agent_result.safe_fallback_used,
        case_state=result.case_state,
    )


def _sse(event: str, data: dict) -> str:
    """编码一个 SSE event。data 始终为单行 JSON，避免多行解析歧义。"""
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"), default=str)
    return f"event: {event}\ndata: {payload}\n\n"


def _chunks(text: str, size: int = 18):
    for index in range(0, len(text), size):
        yield text[index : index + size]


def create_app(
    *,
    settings: Settings | None = None,
    orchestrator: AgentOrchestrator | None = None,
    trace_store: TraceStore | None = None,
) -> FastAPI:
    settings = settings or Settings.from_env()
    orchestrator = orchestrator or build_orchestrator(settings)
    trace_store = trace_store or build_trace_store(settings)

    app = FastAPI(
        title="CampusMind API",
        default_response_class=UTF8JSONResponse,
        version=APP_VERSION,
        description=(
            "Risk-aware CampusMind Agent demo：Redis state/trace、Chroma Vector RAG、"
            "Context-aware Follow-up，以及 SSE Agent event stream + Vue Demo。"
            "资源内容仍为演示知识库，不代表真实学校政策。"
        ),
    )
    app.state.settings = settings
    app.state.orchestrator = orchestrator
    app.state.trace_store = trace_store

    @app.get("/", include_in_schema=False)
    async def root():
        return RedirectResponse(url="/demo")

    @app.get("/demo", response_class=HTMLResponse, include_in_schema=False)
    async def demo() -> HTMLResponse:
        page = WEB_DIR / "index.html"
        if not page.exists():
            raise HTTPException(status_code=404, detail="Demo UI not found")
        return HTMLResponse(page.read_text(encoding="utf-8"))

    @app.get("/health", response_model=HealthResponse, tags=["system"])
    async def health() -> HealthResponse:
        return HealthResponse(
            version=APP_VERSION,
            llm_mode=app.state.settings.llm_mode,
            state_backend=app.state.settings.state_backend,
            trace_backend=app.state.settings.trace_backend,
            rag_backend=app.state.settings.rag_backend,
        )

    @app.post("/chat", response_model=ChatResponse, tags=["chat"])
    async def chat(request: ChatRequest) -> ChatResponse:
        request_id = f"req_{uuid4().hex}"
        started = perf_counter()

        try:
            result = await app.state.orchestrator.handle(request)
        except Exception as exc:  # API 边界兜底；业务内部仍优先使用结构化 fallback。
            latency_ms = (perf_counter() - started) * 1000
            app.state.trace_store.save(
                TraceRecord(
                    request_id=request_id,
                    status="error",
                    latency_ms=latency_ms,
                    error_type=exc.__class__.__name__,
                    error=str(exc)[:500],
                )
            )
            raise HTTPException(
                status_code=500,
                detail={
                    "request_id": request_id,
                    "message": "CampusMind 请求处理失败，请通过 request_id 查询 Trace。",
                },
            ) from exc

        latency_ms = (perf_counter() - started) * 1000
        app.state.trace_store.save(_trace_record(request_id, result, latency_ms))
        return _chat_response(request_id, result)

    @app.post("/chat/stream", tags=["chat"])
    async def chat_stream(request: ChatRequest) -> StreamingResponse:
        """SSE Agent event stream。

        当前语义：先发送 accepted / stage；Orchestrator 完成后发送 meta，随后将最终允许发送的
        response 按 chunk 增量推给前端，最后 done。它是真实 SSE，但不是 LLM token-level stream。
        """
        request_id = f"req_{uuid4().hex}"

        async def event_generator():
            started = perf_counter()
            yield _sse("accepted", {"request_id": request_id})
            yield _sse("stage", {"name": "orchestrating", "label": "Intent / Risk / Policy / Agent Runtime"})
            await asyncio.sleep(0)

            try:
                result = await app.state.orchestrator.handle(request)
            except Exception as exc:
                latency_ms = (perf_counter() - started) * 1000
                app.state.trace_store.save(
                    TraceRecord(
                        request_id=request_id,
                        status="error",
                        latency_ms=latency_ms,
                        error_type=exc.__class__.__name__,
                        error=str(exc)[:500],
                    )
                )
                yield _sse(
                    "error",
                    {
                        "request_id": request_id,
                        "message": "CampusMind 请求处理失败，请通过 request_id 查询 Trace。",
                        "error_type": exc.__class__.__name__,
                    },
                )
                return

            latency_ms = (perf_counter() - started) * 1000
            app.state.trace_store.save(_trace_record(request_id, result, latency_ms))
            response = _chat_response(request_id, result)

            yield _sse(
                "meta",
                {
                    "request_id": request_id,
                    "intent": response.intent.value,
                    "risk": response.risk.value,
                    "selected_agent": response.selected_agent.value,
                    "tools_used": response.tools_used,
                    "guard_passed": response.guard_passed,
                    "guard_violations": response.guard_violations,
                    "safe_fallback_used": response.safe_fallback_used,
                    "latency_ms": round(latency_ms, 2),
                },
            )

            # 这里是“最终回答分块 SSE”，不是模型原始 token stream。
            for chunk in _chunks(response.response):
                yield _sse("chunk", {"text": chunk})
                await asyncio.sleep(0)

            yield _sse(
                "done",
                {
                    "request_id": request_id,
                    "case_state": response.case_state.model_dump(mode="json"),
                    "trace_url": f"/traces/{request_id}",
                },
            )

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    @app.get("/traces/{request_id}", response_model=TraceRecord, tags=["observability"])
    async def get_trace(request_id: str) -> TraceRecord:
        trace = app.state.trace_store.get(request_id)
        if trace is None:
            raise HTTPException(status_code=404, detail="Trace not found")
        return trace

    @app.get("/traces", response_model=list[TraceRecord], tags=["observability"])
    async def list_traces(
        limit: int = Query(default=20, ge=1, le=100),
    ) -> list[TraceRecord]:
        return app.state.trace_store.recent(limit=limit)

    return app


app = create_app()
