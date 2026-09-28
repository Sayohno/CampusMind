"""CampusMind 多 Agent 路由与 Orchestrator。"""
from __future__ import annotations

from agents.base_agent import BaseAgent
from core.intent_analyzer import IntentAnalyzer
from core.risk_analyzer import RiskAnalyzer
from core.risk_policy import RiskPolicyEngine
from core.schemas import (
    AgentName,
    AgentRequest,
    CaseStatus,
    ChatRequest,
    GuardResult,
    OrchestratorResult,
    PolicyDecision,
)
from safety.response_guard import ResponseGuard
from state.case_store import CaseStore
from state.memory_store import ConversationMemory


class Router:
    """Thin Router：只把 Policy 已有决策映射到具体 Agent。"""

    def __init__(self, agents: dict[AgentName, BaseAgent]) -> None:
        self.agents = agents

    def route(self, policy: PolicyDecision) -> BaseAgent:
        try:
            return self.agents[policy.allowed_agent]
        except KeyError as exc:
            raise RuntimeError(f"未注册 Agent: {policy.allowed_agent.value}") from exc


class AgentOrchestrator:
    def __init__(
        self,
        *,
        intent_analyzer: IntentAnalyzer,
        risk_analyzer: RiskAnalyzer,
        policy_engine: RiskPolicyEngine,
        router: Router,
        memory_store: ConversationMemory,
        case_store: CaseStore,
        response_guard: ResponseGuard | None = None,
        memory_limit: int = 8,
    ) -> None:
        self.intent_analyzer = intent_analyzer
        self.risk_analyzer = risk_analyzer
        self.policy_engine = policy_engine
        self.router = router
        self.memory_store = memory_store
        self.case_store = case_store
        self.response_guard = response_guard or ResponseGuard()
        self.memory_limit = memory_limit

    async def handle(self, request: ChatRequest) -> OrchestratorResult:
        # 1) 读取 Conversation Memory。
        memory_history = self.memory_store.get_recent(
            user_id=request.user_id,
            conversation_id=request.conversation_id,
            limit=self.memory_limit,
        )

        # 2) 结构化理解与确定性 Policy。
        intent_result = self.intent_analyzer.analyze(request.message)
        risk_assessment = self.risk_analyzer.analyze(request.message)
        policy = self.policy_engine.decide(intent_result, risk_assessment)

        # 3) CaseState 是当前结构化快照。风险等级由 RiskAnalyzer 写入，不让 LLM 自行覆盖。
        case_state = self.case_store.ensure(
            user_id=request.user_id,
            conversation_id=request.conversation_id,
            topic=intent_result.intent.value,
            risk_level=risk_assessment.risk_level,
        )
        if policy.force_escalation:
            case_state.status = CaseStatus.ESCALATED
            case_state = self.case_store.save(case_state)

        # 4) Thin Router 选 Agent，并注入 Memory + CaseState。
        agent = self.router.route(policy)
        agent_request = AgentRequest(
            user_id=request.user_id,
            conversation_id=request.conversation_id,
            message=request.message,
            intent_result=intent_result,
            risk_assessment=risk_assessment,
            policy=policy,
            memory_history=memory_history,
            case_state=case_state,
        )
        candidate_result = await agent.run(agent_request)

        # 5) Response Guard 管出口：先检查候选回复，再决定是否直接发送。
        guard = self.response_guard.evaluate(agent_request, candidate_result)
        if guard.passed:
            agent_result = candidate_result.model_copy(
                update={
                    "guard_result": guard,
                    "safe_fallback_used": False,
                }
            )
        else:
            fallback = self.response_guard.safe_fallback(agent_request, guard)
            guard_with_fallback = GuardResult(
                passed=False,
                violations=guard.violations,
                safe_fallback_used=True,
            )
            agent_result = candidate_result.model_copy(
                update={
                    "candidate_response": candidate_result.response,
                    "response": fallback,
                    "guard_result": guard_with_fallback,
                    "safe_fallback_used": True,
                }
            )

        # 6) 只把最终允许发送的回复写入 Memory，不把 Guard 拦截的候选文本当成正式回复。
        self.memory_store.append_turn(
            user_id=request.user_id,
            conversation_id=request.conversation_id,
            user_message=request.message,
            assistant_message=agent_result.response,
        )

        # Tool 可能在 Runtime 中更新 CaseState，因此最后重新读取最新快照。
        latest_case = self.case_store.get(
            user_id=request.user_id,
            conversation_id=request.conversation_id,
        ) or case_state
        latest_case.current_risk_level = risk_assessment.risk_level
        latest_case.topic = intent_result.intent.value
        if policy.force_escalation:
            latest_case.status = CaseStatus.ESCALATED
        latest_case = self.case_store.save(latest_case)

        # 本轮至少完成了确定性风险/主题状态持久化，因此 CaseState 已更新。
        agent_result = agent_result.model_copy(update={"case_updated": True})
        memory_size = self.memory_store.count(
            user_id=request.user_id,
            conversation_id=request.conversation_id,
        )

        return OrchestratorResult(
            intent_result=intent_result,
            risk_assessment=risk_assessment,
            policy_decision=policy,
            selected_agent=agent.name,
            agent_result=agent_result,
            case_state=latest_case,
            memory_size=memory_size,
        )
