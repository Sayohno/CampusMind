"""CampusMind BaseAgent 与 Agent Runtime Loop。"""
from __future__ import annotations

import json
from typing import Any

from agents.tools import ToolError, ToolRegistry
from core.llm_client import BaseLLMClient, LLMClientError
from core.schemas import AgentName, AgentRequest, AgentResult, ToolCall, ToolExecution


class BaseAgent:
    """共享 Runtime + 不同 Role / Prompt / Tool Scope。"""

    def __init__(
        self,
        *,
        name: AgentName,
        system_prompt: str,
        llm_client: BaseLLMClient,
        tool_registry: ToolRegistry,
        allowed_tools: set[str] | None = None,
        max_rounds: int = 4,
    ) -> None:
        if max_rounds < 1:
            raise ValueError("max_rounds 必须 >= 1")
        self.name = name
        self.system_prompt = system_prompt
        self.llm_client = llm_client
        self.tool_registry = tool_registry
        self.allowed_tools = set(allowed_tools or set())
        self.max_rounds = max_rounds

    def build_messages(self, request: AgentRequest) -> list[dict[str, Any]]:
        context = {
            "intent": request.intent_result.model_dump(mode="json"),
            "risk": request.risk_assessment.model_dump(mode="json"),
            "policy": request.policy.model_dump(mode="json"),
            "case_state": (
                request.case_state.model_dump(mode="json")
                if request.case_state is not None
                else None
            ),
        }
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "system",
                "content": "CampusMind structured context:\n"
                + json.dumps(context, ensure_ascii=False),
            },
        ]

        # Conversation Memory 是跨用户轮次历史；这里只装载最近一小段，控制上下文长度。
        for item in request.memory_history:
            messages.append({"role": item.role, "content": item.content})

        messages.append({"role": "user", "content": request.message})
        return messages

    async def run(self, request: AgentRequest) -> AgentResult:
        messages = self.build_messages(request)
        tools_used: list[str] = []
        tool_executions: list[ToolExecution] = []
        executed_cache: dict[str, Any] = {}
        llm_tools = self.tool_registry.llm_tools(self.allowed_tools)

        # 高风险 mandatory_tools 不依赖 LLM 是否“想起来”。Runtime 在第一次模型调用前强制执行。
        mandatory_results, mandatory_error = await self._execute_mandatory_tools(
            request=request,
            tools_used=tools_used,
            tool_executions=tool_executions,
            executed_cache=executed_cache,
        )
        if mandatory_results:
            messages.insert(
                -1,
                {
                    "role": "system",
                    "content": (
                        "CampusMind mandatory tool results (deterministic runtime):\n"
                        + json.dumps(mandatory_results, ensure_ascii=False, default=str)
                        + "\n必须基于这些结果回答；不得声称执行了结果中没有发生的外部动作。"
                    ),
                },
            )

        # 必做安全动作失败时，不继续放开普通生成。
        if mandatory_error is not None:
            return AgentResult(
                response=(
                    "当前安全处理所需的必要能力暂时不可用。"
                    "请不要依赖本系统完成紧急处置；如果你正处于现实中的紧急危险，"
                    "请优先联系身边可信赖的人、所在学校正式支持渠道或当地紧急服务。"
                ),
                tools_used=tools_used,
                tool_executions=tool_executions,
                case_updated=("upsert_support_case" in tools_used),
                handoff_requested=(self.name == AgentName.ESCALATION),
                runtime_rounds=0,
            )

        for round_index in range(1, self.max_rounds + 1):
            try:
                response = await self.llm_client.chat(messages, tools=llm_tools)
            except LLMClientError as exc:
                return AgentResult(
                    response=f"模型调用暂时失败：{exc}",
                    tools_used=tools_used,
                    tool_executions=tool_executions,
                    handoff_requested=(self.name == AgentName.ESCALATION),
                    runtime_rounds=round_index,
                )

            if not response.tool_calls:
                final_text = (response.content or "").strip()
                if not final_text:
                    final_text = "模型没有返回可用文本，请稍后重试。"
                return AgentResult(
                    response=final_text,
                    tools_used=tools_used,
                    tool_executions=tool_executions,
                    case_updated=("upsert_support_case" in tools_used),
                    handoff_requested=(self.name == AgentName.ESCALATION),
                    runtime_rounds=round_index,
                )

            self._append_assistant_tool_calls(messages, response.content, response.tool_calls)

            for tool_call in response.tool_calls:
                tool_result = await self._execute_tool(
                    tool_call,
                    request,
                    tools_used,
                    tool_executions,
                    executed_cache,
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_call.name,
                        "content": json.dumps(tool_result, ensure_ascii=False, default=str),
                    }
                )

        return AgentResult(
            response="Agent Runtime 达到最大工具调用轮数，已停止继续执行。",
            tools_used=tools_used,
            tool_executions=tool_executions,
            case_updated=("upsert_support_case" in tools_used),
            handoff_requested=(self.name == AgentName.ESCALATION),
            runtime_rounds=self.max_rounds,
        )

    async def _execute_mandatory_tools(
        self,
        *,
        request: AgentRequest,
        tools_used: list[str],
        tool_executions: list[ToolExecution],
        executed_cache: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
        results: list[dict[str, Any]] = []
        for tool_name in request.policy.mandatory_tools:
            arguments = self._mandatory_arguments(tool_name, request)
            signature = self._tool_signature(tool_name, arguments)
            try:
                result = await self.tool_registry.execute(
                    tool_name=tool_name,
                    arguments=arguments,
                    request=request,
                    allowed_tools=self.allowed_tools,
                )
            except ToolError as exc:
                error = {
                    "tool": tool_name,
                    "ok": False,
                    "error_type": exc.__class__.__name__,
                    "error": str(exc),
                }
                tool_executions.append(
                    ToolExecution(
                        name=tool_name,
                        source="mandatory",
                        arguments=arguments,
                        status="failed",
                        error=str(exc),
                    )
                )
                results.append(error)
                return results, error

            executed_cache[signature] = result
            tools_used.append(tool_name)
            tool_executions.append(
                ToolExecution(
                    name=tool_name,
                    source="mandatory",
                    arguments=arguments,
                    status="executed",
                    result=result,
                )
            )
            results.append({"tool": tool_name, "ok": True, "result": result})

        return results, None

    @staticmethod
    def _mandatory_arguments(tool_name: str, request: AgentRequest) -> dict[str, Any]:
        """Policy 只指定“必须做什么”；Runtime 负责提供确定性参数。"""
        if tool_name == "search_verified_resources":
            return {"query": request.message, "top_k": 3}
        if tool_name == "upsert_support_case":
            return {
                "status": "escalated",
                "add_known_context": ["high_risk_policy_path_activated"],
            }
        return {}

    async def _execute_tool(
        self,
        tool_call: ToolCall,
        request: AgentRequest,
        tools_used: list[str],
        tool_executions: list[ToolExecution],
        executed_cache: dict[str, Any],
    ) -> dict[str, Any]:
        signature = self._tool_signature(tool_call.name, tool_call.arguments)

        # 只跳过“工具名 + 参数都相同”的重复执行。相同 Tool 但参数不同仍允许再次执行，
        # 例如 mandatory upsert 负责标记 escalated，LLM 后续可用另一组参数补 actions_done。
        if signature in executed_cache:
            cached = executed_cache[signature]
            tool_executions.append(
                ToolExecution(
                    name=tool_call.name,
                    source="llm",
                    arguments=tool_call.arguments,
                    status="skipped_duplicate",
                    result=cached,
                )
            )
            return {
                "ok": True,
                "skipped": True,
                "reason": "duplicate_tool_call",
                "result": cached,
            }

        try:
            result = await self.tool_registry.execute(
                tool_name=tool_call.name,
                arguments=tool_call.arguments,
                request=request,
                allowed_tools=self.allowed_tools,
            )
        except ToolError as exc:
            tool_executions.append(
                ToolExecution(
                    name=tool_call.name,
                    source="llm",
                    arguments=tool_call.arguments,
                    status="failed",
                    error=str(exc),
                )
            )
            return {
                "ok": False,
                "error_type": exc.__class__.__name__,
                "error": str(exc),
            }

        executed_cache[signature] = result
        tools_used.append(tool_call.name)
        tool_executions.append(
            ToolExecution(
                name=tool_call.name,
                source="llm",
                arguments=tool_call.arguments,
                status="executed",
                result=result,
            )
        )
        return {"ok": True, "result": result}

    @staticmethod
    def _tool_signature(tool_name: str, arguments: dict[str, Any]) -> str:
        normalized = json.dumps(
            arguments,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        return f"{tool_name}:{normalized}"

    @staticmethod
    def _append_assistant_tool_calls(
        messages: list[dict[str, Any]],
        content: str | None,
        tool_calls: list[ToolCall],
    ) -> None:
        messages.append(
            {
                "role": "assistant",
                "content": content,
                "tool_calls": [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": json.dumps(call.arguments, ensure_ascii=False),
                        },
                    }
                    for call in tool_calls
                ],
            }
        )


class GuidanceAgent(BaseAgent):
    pass


class SupportAgent(BaseAgent):
    pass


class ResourceAgent(BaseAgent):
    pass


class EscalationAgent(BaseAgent):
    pass


class GeneralAgent(BaseAgent):
    pass
