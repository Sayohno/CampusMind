"""CampusMind FastAPI 入口。

正式服务主链：HTTP /chat -> Orchestrator -> Agent Runtime -> Guard -> Response。
当前 Memory / CaseState / Trace 使用进程内存储，容器重启后会清空；这是当前 Demo 边界。
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
from observability.trace_store import InMemoryTraceStore
from rag.resource_store import VerifiedResourceStore
from safety.response_guard import ResponseGuard
from state.case_store import CaseStore, InMemoryCaseStore
from state.memory_store import ConversationMemory, InMemoryConversationMemory

APP_VERSION = "0.9.0"


class UTF8JSONResponse(JSONResponse):
    """为 Windows PowerShell 5.1 等旧客户端显式声明 UTF-8。"""

    media_type = "application/json; charset=utf-8"


def build_orchestrator(
    settings: Settings | None = None,
    *,
    llm_client: BaseLLMClient | None = None,
    memory_store: ConversationMemory | None = None,
    case_store: CaseStore | None = None,
    resource_store: VerifiedResourceStore | None = None,
) -> AgentOrchestrator:
    """构建一套独立的 CampusMind Orchestrator。

    默认使用 Settings() 的 static 模式，确保单元测试不会因为本机 .env=real 而误调真实 API。
    FastAPI 正式启动时会显式传入 Settings.from_env()。
    """
    settings = settings or Settings()
    memory_store = memory_store or InMemoryConversationMemory()
    case_store = case_store or InMemoryCaseStore()
    resource_store = resource_store or VerifiedResourceStore.demo()
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
    trace_store: InMemoryTraceStore | None = None,
) -> FastAPI:
    settings = settings or Settings.from_env()
    orchestrator = orchestrator or build_orchestrator(settings)
    trace_store = trace_store or InMemoryTraceStore()

    app = FastAPI(
        title="CampusMind API",
        default_response_class=UTF8JSONResponse,
        version=APP_VERSION,
        description=(
            "Risk-aware CampusMind Agent demo. "
            "当前资源库、Memory、CaseState 与 Trace 均为演示/进程内实现。"
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
