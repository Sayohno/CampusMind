"""CampusMind 内部数据契约。

所有模块通过稳定的 Pydantic Schema 交换数据，避免临时 dict / 字符串
把模块之间耦合在一起。
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class IntentType(str, Enum):
    ACADEMIC_CAREER = "academic_career"
    EMOTIONAL_SUPPORT = "emotional_support"
    RELATIONSHIP = "relationship"
    RESOURCE_QUERY = "resource_query"
    HUMAN_SUPPORT = "human_support"
    GENERAL = "general"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class AgentName(str, Enum):
    GUIDANCE = "GuidanceAgent"
    SUPPORT = "SupportAgent"
    RESOURCE = "ResourceAgent"
    ESCALATION = "EscalationAgent"
    GENERAL = "GeneralAgent"


class CaseStatus(str, Enum):
    ACTIVE = "active"
    ESCALATED = "escalated"
    RESOLVED = "resolved"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ChatRequest(BaseModel):
    user_id: str = Field(min_length=1)
    conversation_id: str = Field(min_length=1)
    message: str = Field(min_length=1)


class ConversationMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1)
    created_at: datetime = Field(default_factory=utc_now)


class IntentResult(BaseModel):
    intent: IntentType
    confidence: float = Field(ge=0.0, le=1.0)


class RiskAssessment(BaseModel):
    risk_level: RiskLevel
    signals: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


class PolicyDecision(BaseModel):
    autonomy_level: str
    allowed_agent: AgentName
    mandatory_tools: list[str] = Field(default_factory=list)
    allow_general_advice: bool = True
    force_escalation: bool = False


class SupportCaseState(BaseModel):
    """当前支持任务的结构化快照，不是医疗病历。"""

    case_id: str
    user_id: str
    conversation_id: str
    topic: str = "general"
    goal: str | None = None
    known_context: list[str] = Field(default_factory=list)
    actions_done: list[str] = Field(default_factory=list)
    next_actions: list[str] = Field(default_factory=list)
    current_risk_level: RiskLevel = RiskLevel.LOW
    resource_refs: list[str] = Field(default_factory=list)
    status: CaseStatus = CaseStatus.ACTIVE
    updated_at: datetime = Field(default_factory=utc_now)


class AgentRequest(BaseModel):
    user_id: str
    conversation_id: str
    message: str
    intent_result: IntentResult
    risk_assessment: RiskAssessment
    policy: PolicyDecision
    memory_history: list[ConversationMessage] = Field(default_factory=list)
    case_state: SupportCaseState | None = None


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class LLMResponse(BaseModel):
    content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)


class ToolExecution(BaseModel):
    """一次真实 Tool 执行/跳过/失败的可观测记录。"""

    name: str
    source: Literal["mandatory", "llm"]
    arguments: dict[str, Any] = Field(default_factory=dict)
    status: Literal["executed", "skipped_duplicate", "failed"]
    result: Any | None = None
    error: str | None = None


class GuardResult(BaseModel):
    """Response Guard 的确定性检查结果。"""

    passed: bool
    violations: list[str] = Field(default_factory=list)
    safe_fallback_used: bool = False


class AgentResult(BaseModel):
    response: str
    tools_used: list[str] = Field(default_factory=list)
    tool_executions: list[ToolExecution] = Field(default_factory=list)
    case_updated: bool = False
    handoff_requested: bool = False
    runtime_rounds: int = 0
    candidate_response: str | None = None
    guard_result: GuardResult | None = None
    safe_fallback_used: bool = False


class OrchestratorResult(BaseModel):
    intent_result: IntentResult
    risk_assessment: RiskAssessment
    policy_decision: PolicyDecision
    selected_agent: AgentName
    agent_result: AgentResult
    case_state: SupportCaseState
    memory_size: int = 0


class ToolTraceItem(BaseModel):
    """对外可观测 Tool Trace：只记录参数键，不记录敏感参数值。"""

    name: str
    source: Literal["mandatory", "llm"]
    status: Literal["executed", "skipped_duplicate", "failed"]
    argument_keys: list[str] = Field(default_factory=list)
    error: str | None = None


class TraceRecord(BaseModel):
    """单次 /chat 请求的结构化 Trace。"""

    request_id: str
    created_at: datetime = Field(default_factory=utc_now)
    status: Literal["success", "error"]
    intent: IntentType | None = None
    risk: RiskLevel | None = None
    allowed_agent: AgentName | None = None
    selected_agent: AgentName | None = None
    tool_executions: list[ToolTraceItem] = Field(default_factory=list)
    guard_passed: bool | None = None
    guard_violations: list[str] = Field(default_factory=list)
    safe_fallback_used: bool = False
    memory_size: int | None = None
    case_status: CaseStatus | None = None
    latency_ms: float = Field(ge=0.0)
    error_type: str | None = None
    error: str | None = None


class ChatResponse(BaseModel):
    request_id: str
    response: str
    intent: IntentType
    risk: RiskLevel
    selected_agent: AgentName
    tools_used: list[str] = Field(default_factory=list)
    guard_passed: bool | None = None
    guard_violations: list[str] = Field(default_factory=list)
    safe_fallback_used: bool = False
    case_state: SupportCaseState


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    service: str = "CampusMind"
    version: str
    llm_mode: str
    state_backend: str = "in_memory"
    trace_backend: str = "in_memory"
