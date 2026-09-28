"""CampusMind FastAPI 入口。

V1.1：HTTP /chat -> Orchestrator -> Agent Runtime -> Guard -> Response，
并支持 Redis 状态/Trace 后端与 Chroma Vector RAG。单元测试默认仍使用 in-memory + keyword，
Docker Compose 默认启用 Redis + Chroma。
"""
from __future__ import annotations

from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse

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

APP_VERSION = "1.1.1"


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
            "Risk-aware CampusMind Agent demo. V1.1 支持 Redis state/trace backend 与 "
            "Chroma Vector RAG；资源内容仍为演示知识库，不代表真实学校政策。"
        ),
    )
    app.state.settings = settings
    app.state.orchestrator = orchestrator
    app.state.trace_store = trace_store

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
        guard = result.agent_result.guard_result
        tool_trace = [
            ToolTraceItem(
                name=item.name,
                source=item.source,
                status=item.status,
                argument_keys=sorted(item.arguments.keys()),
                error=item.error,
            )
            for item in result.agent_result.tool_executions
        ]
        trace = TraceRecord(
            request_id=request_id,
            status="success",
            intent=result.intent_result.intent,
            risk=result.risk_assessment.risk_level,
            allowed_agent=result.policy_decision.allowed_agent,
            selected_agent=result.selected_agent,
            tool_executions=tool_trace,
            guard_passed=(guard.passed if guard is not None else None),
            guard_violations=(guard.violations if guard is not None else []),
            safe_fallback_used=result.agent_result.safe_fallback_used,
            memory_size=result.memory_size,
            case_status=result.case_state.status,
            latency_ms=latency_ms,
        )
        app.state.trace_store.save(trace)

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
