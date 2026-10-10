"""Route one request through analysis, policy, runtime and output guard."""
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
    RiskAssessment,
    RiskLevel,
)
from core.semantic_analyzer import SemanticAnalyzer
from safety.response_guard import ResponseGuard
from state.case_store import CaseStore
from state.memory_store import ConversationMemory


class Router:
    """Map an already-decided policy target to a configured agent."""

    def __init__(self, agents: dict[AgentName, BaseAgent]) -> None:
        self.agents = agents

    def route(self, policy: PolicyDecision) -> BaseAgent:
        try:
            return self.agents[policy.allowed_agent]
        except KeyError as exc:
            raise RuntimeError(
                f"未注册 Agent: {policy.allowed_agent.value}"
            ) from exc


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
        semantic_analyzer: SemanticAnalyzer | None = None,
        memory_limit: int = 8,
    ) -> None:
        self.intent_analyzer = intent_analyzer
        self.risk_analyzer = risk_analyzer
        self.policy_engine = policy_engine
        self.router = router
        self.memory_store = memory_store
        self.case_store = case_store
        self.response_guard = response_guard or ResponseGuard()
        self.semantic_analyzer = semantic_analyzer
        self.memory_limit = memory_limit

    async def handle(self, request: ChatRequest) -> OrchestratorResult:
        memory_history = self.memory_store.get_recent(
            user_id=request.user_id,
            conversation_id=request.conversation_id,
            limit=self.memory_limit,
        )
        previous_case = self.case_store.get(
            user_id=request.user_id,
            conversation_id=request.conversation_id,
        )

        intent_result = self.intent_analyzer.analyze(
            request.message,
            memory_history=memory_history,
            case_topic=previous_case.topic if previous_case is not None else None,
        )
        risk_assessment = self.risk_analyzer.analyze(request.message)

        intent_needs_semantic = self.intent_analyzer.needs_semantic_fallback(
            request.message,
            intent_result,
        )
        risk_needs_semantic = self.risk_analyzer.needs_semantic_fallback(
            request.message,
            risk_assessment,
        )

        if self.semantic_analyzer is not None and (
            intent_needs_semantic or risk_needs_semantic
        ):
            try:
                semantic_result = await self.semantic_analyzer.analyze(
                    request.message
                )
            except Exception:
                # Fail closed only when fallback failed around a potential high-risk phrase.
                if (
                    risk_needs_semantic
                    and self.risk_analyzer.has_potential_high_signal(
                        request.message
                    )
                ):
                    risk_assessment = RiskAssessment(
                        risk_level=RiskLevel.HIGH,
                        signals=[
                            "semantic_fallback_failed",
                            "potential_high_risk_signal",
                        ],
                        confidence=0.80,
                    )
            else:
                if (
                    intent_needs_semantic
                    and semantic_result.intent_result.confidence >= 0.70
                ):
                    intent_result = semantic_result.intent_result

                if (
                    risk_needs_semantic
                    and semantic_result.risk_assessment.confidence >= 0.75
                ):
                    semantic_risk = semantic_result.risk_assessment
                    deterministic_high = (
                        risk_assessment.risk_level == RiskLevel.HIGH
                    )
                    semantic_downgrade = (
                        semantic_risk.risk_level != RiskLevel.HIGH
                    )
                    if not (deterministic_high and semantic_downgrade):
                        risk_assessment = semantic_risk

        # Policy remains deterministic even when semantic fallback is used.
        policy = self.policy_engine.decide(intent_result, risk_assessment)

        case_state = self.case_store.ensure(
            user_id=request.user_id,
            conversation_id=request.conversation_id,
            topic=intent_result.intent.value,
            risk_level=risk_assessment.risk_level,
        )
        if policy.force_escalation:
            case_state.status = CaseStatus.ESCALATED
            case_state = self.case_store.save(case_state)

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
        guard = self.response_guard.evaluate(agent_request, candidate_result)

        if guard.passed:
            agent_result = candidate_result.model_copy(
                update={
                    "guard_result": guard,
                    "safe_fallback_used": False,
                }
            )
        else:
            fallback = self.response_guard.safe_fallback(
                agent_request,
                guard,
            )
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

        # Persist only the response that passed the output boundary.
        self.memory_store.append_turn(
            user_id=request.user_id,
            conversation_id=request.conversation_id,
            user_message=request.message,
            assistant_message=agent_result.response,
        )

        latest_case = self.case_store.get(
            user_id=request.user_id,
            conversation_id=request.conversation_id,
        ) or case_state
        latest_case.current_risk_level = risk_assessment.risk_level
        latest_case.topic = intent_result.intent.value
        if policy.force_escalation:
            latest_case.status = CaseStatus.ESCALATED
        latest_case = self.case_store.save(latest_case)

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
