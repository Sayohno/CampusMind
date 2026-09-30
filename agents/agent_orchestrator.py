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
    RiskAssessment,
    RiskLevel,
)
from core.semantic_analyzer import SemanticAnalyzer
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

    async def handle(
        self,
        request: ChatRequest,
    ) -> OrchestratorResult:
        # 1) 读取 Conversation Memory。
        memory_history = self.memory_store.get_recent(
            user_id=request.user_id,
            conversation_id=request.conversation_id,
            limit=self.memory_limit,
        )

        # 2) 读取上一轮 CaseState。
        #
        # 对“继续刚才的问题”这类 follow-up，
        # IntentAnalyzer 只读取最小必要上下文：
        # recent user memory + previous case topic。
        previous_case = self.case_store.get(
            user_id=request.user_id,
            conversation_id=request.conversation_id,
        )

        # 3) Deterministic Fast Path：
        #    先使用规则完成 Intent / Risk 分析。
        intent_result = self.intent_analyzer.analyze(
            request.message,
            memory_history=memory_history,
            case_topic=(
                previous_case.topic
                if previous_case is not None
                else None
            ),
        )

        risk_assessment = self.risk_analyzer.analyze(
            request.message
        )

        # 4) 判断规则结果是否存在明显语义不确定性。
        intent_needs_semantic = (
            self.intent_analyzer.needs_semantic_fallback(
                request.message,
                intent_result,
            )
        )

        risk_needs_semantic = (
            self.risk_analyzer.needs_semantic_fallback(
                request.message,
                risk_assessment,
            )
        )

        # 5) Semantic Fallback：
        #
        # 只有：
        # - SemanticAnalyzer 已启用；
        # - 并且 Intent / Risk 至少一个存在明显不确定性；
        #
        # 才调用一次 LLM。
        #
        # SemanticAnalyzer 只负责“理解”。
        # 它无权决定 Agent、Tool Scope、mandatory actions
        # 或 escalation policy。
        if (
            self.semantic_analyzer is not None
            and (
                intent_needs_semantic
                or risk_needs_semantic
            )
        ):
            try:
                semantic_result = (
                    await self.semantic_analyzer.analyze(
                        request.message
                    )
                )

            except Exception:
                # Semantic fallback 属于增强能力，
                # 模型/API/解析失败不能让整个业务主流程崩溃。
                #
                # 大多数情况下继续保留 deterministic result。
                #
                # 但如果 RiskAnalyzer 已发现潜在 high-risk phrase，
                # Semantic 又恰好失败，则安全侧 fail-closed：
                # 将风险提升为 HIGH，由后面的确定性 PolicyEngine
                # 强制进入 Escalation。
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
                # Intent：
                # 只有本轮确实需要 Semantic Fallback，
                # 并且模型置信度足够时才覆盖规则结果。
                if (
                    intent_needs_semantic
                    and semantic_result.intent_result.confidence
                    >= 0.70
                ):
                    intent_result = (
                        semantic_result.intent_result
                    )

                # Risk：
                # Semantic 可以：
                # - 修正 hard negative；
                # - 将 LOW/MEDIUM 升为更高风险；
                # - 对模糊语境做消歧。
                #
                # 但如果确定性 RiskAnalyzer 已经给出 HIGH，
                # SemanticAnalyzer 不允许把 HIGH 降级。
                if (
                    risk_needs_semantic
                    and semantic_result.risk_assessment.confidence
                    >= 0.75
                ):
                    semantic_risk = (
                        semantic_result.risk_assessment
                    )

                    if (
                        risk_assessment.risk_level
                        == RiskLevel.HIGH
                        and semantic_risk.risk_level
                        != RiskLevel.HIGH
                    ):
                        pass
                    else:
                        risk_assessment = semantic_risk

        # 6) 到这里语义理解结束。
        #
        # 真正的安全政策仍然只由确定性的
        # RiskPolicyEngine 决定。
        policy = self.policy_engine.decide(
            intent_result,
            risk_assessment,
        )

        # 7) CaseState 是当前结构化业务快照。
        #
        # 风险等级来自 RiskAnalyzer / Semantic fallback
        # 最终得到的 RiskAssessment。
        #
        # LLM 无法直接修改 CaseState 风险等级。
        case_state = self.case_store.ensure(
            user_id=request.user_id,
            conversation_id=request.conversation_id,
            topic=intent_result.intent.value,
            risk_level=risk_assessment.risk_level,
        )

        if policy.force_escalation:
            case_state.status = CaseStatus.ESCALATED
            case_state = self.case_store.save(case_state)

        # 8) Thin Router 只执行 Policy 已经做出的 Agent 决策。
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

        # 9) Agent Runtime。
        candidate_result = await agent.run(
            agent_request
        )

        # 10) Response Guard 管出口。
        #
        # Agent 生成的是 candidate response。
        # Guard 决定它是否能够真正发送给用户。
        guard = self.response_guard.evaluate(
            agent_request,
            candidate_result,
        )

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
                    "candidate_response":
                        candidate_result.response,
                    "response": fallback,
                    "guard_result": guard_with_fallback,
                    "safe_fallback_used": True,
                }
            )

        # 11) Memory 只保存最终允许发送的响应。
        #
        # 被 Guard 拦截的 candidate response
        # 不进入正式 Conversation Memory。
        self.memory_store.append_turn(
            user_id=request.user_id,
            conversation_id=request.conversation_id,
            user_message=request.message,
            assistant_message=agent_result.response,
        )

        # 12) Tool Runtime 可能已经更新过 CaseState，
        # 因此重新读取最新快照。
        latest_case = self.case_store.get(
            user_id=request.user_id,
            conversation_id=request.conversation_id,
        ) or case_state

        latest_case.current_risk_level = (
            risk_assessment.risk_level
        )

        latest_case.topic = (
            intent_result.intent.value
        )

        if policy.force_escalation:
            latest_case.status = CaseStatus.ESCALATED

        latest_case = self.case_store.save(
            latest_case
        )

        agent_result = agent_result.model_copy(
            update={
                "case_updated": True,
            }
        )

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