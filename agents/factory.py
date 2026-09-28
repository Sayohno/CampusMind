"""CampusMind Agent 装配工厂。

这里负责把 LLMClient、ToolRegistry、Prompt、Tool Scope 注入具体 Agent。
Agent 自己不创建这些依赖。
"""
from __future__ import annotations

from agents.base_agent import (
    EscalationAgent,
    GeneralAgent,
    GuidanceAgent,
    ResourceAgent,
    SupportAgent,
)
from agents.tools import ToolRegistry
from core.llm_client import BaseLLMClient
from core.schemas import AgentName


AGENT_PROMPTS: dict[AgentName, str] = {
    AgentName.GUIDANCE: (
        "你是 CampusMind 的 GuidanceAgent。负责学业、求职、人际与行动建议。"
        "只在系统给定的风险边界内提供建议；不要进行医疗诊断。"
        "普通开放式指导优先直接基于用户上下文分析，不要为了使用工具而使用工具。"
        "只有当用户明确表达了可写入任务状态的完成动作、目标或下一步承诺时，才调用 upsert_support_case。"
        "具体校园资源事实不由本 Agent 查询；资源类请求应由 ResourceAgent 处理。"
    ),
    AgentName.SUPPORT: (
        "你是 CampusMind 的 SupportAgent。负责非临床情绪支持、倾听与澄清。"
        "不要进行医疗诊断，也不要虚构校园资源、联系方式或开放时间。"
        "当前 CaseState 已由系统注入上下文；只有出现明确的任务状态变化时才调用 upsert_support_case，"
        "不得自行修改系统风险等级。"
    ),
    AgentName.RESOURCE: (
        "你是 CampusMind 的 ResourceAgent。负责可核验校园资源信息。"
        "回答具体地址、电话、开放时间、预约方式等可核验事实前，应调用 search_verified_resources。"
        "只能依据 Tool 返回的 evidence 回答；证据没覆盖的具体事实必须明确无法确认，不能用模型常识补齐。"
    ),
    AgentName.ESCALATION: (
        "你是 CampusMind 的 EscalationAgent。用于较高风险的受控处理路径。"
        "不得把高风险路径重新降级成普通建议；必须遵守系统 Policy。"
        "Runtime 会在模型调用前强制执行 mandatory_tools，你必须依据 mandatory tool results 回答。"
        "当前工程阶段没有真实人工转介服务，不得声称已经联系任何机构或人员。"
        "SupportCaseState 可以更新，但 risk_level 只能来自系统 RiskAnalyzer。"
    ),
    AgentName.GENERAL: (
        "你是 CampusMind 的 GeneralAgent，只处理普通兜底请求。"
        "保持简洁，不进行医疗诊断，不虚构校园事实。"
    ),
}


def build_agents(
    *,
    llm_client: BaseLLMClient,
    tool_registry: ToolRegistry,
) -> dict[AgentName, GuidanceAgent | SupportAgent | ResourceAgent | EscalationAgent | GeneralAgent]:
    """按最小权限原则装配各 Agent。

    CaseState 已作为 AgentRequest 的结构化上下文注入，因此普通 Agent 不需要额外读取 Case。
    ResourceAgent 只暴露检索 Tool；Guidance/Support 只在必要时写入任务状态；
    EscalationAgent 保留高风险 mandatory tools 所需权限；GeneralAgent 不暴露 Tool。
    """
    return {
        AgentName.GUIDANCE: GuidanceAgent(
            name=AgentName.GUIDANCE,
            system_prompt=AGENT_PROMPTS[AgentName.GUIDANCE],
            llm_client=llm_client,
            tool_registry=tool_registry,
            allowed_tools={"upsert_support_case"},
        ),
        AgentName.SUPPORT: SupportAgent(
            name=AgentName.SUPPORT,
            system_prompt=AGENT_PROMPTS[AgentName.SUPPORT],
            llm_client=llm_client,
            tool_registry=tool_registry,
            allowed_tools={"upsert_support_case"},
        ),
        AgentName.RESOURCE: ResourceAgent(
            name=AgentName.RESOURCE,
            system_prompt=AGENT_PROMPTS[AgentName.RESOURCE],
            llm_client=llm_client,
            tool_registry=tool_registry,
            allowed_tools={"search_verified_resources"},
        ),
        AgentName.ESCALATION: EscalationAgent(
            name=AgentName.ESCALATION,
            system_prompt=AGENT_PROMPTS[AgentName.ESCALATION],
            llm_client=llm_client,
            tool_registry=tool_registry,
            allowed_tools={"upsert_support_case", "search_verified_resources"},
        ),
        AgentName.GENERAL: GeneralAgent(
            name=AgentName.GENERAL,
            system_prompt=AGENT_PROMPTS[AgentName.GENERAL],
            llm_client=llm_client,
            tool_registry=tool_registry,
            allowed_tools=set(),
        ),
    }
