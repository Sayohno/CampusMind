"""CampusMind Agent Tools 与 ToolRegistry。

Runtime 只通过 Registry 查找、校验和执行工具，不在 Agent 中写大量 if/elif。
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from core.schemas import AgentRequest, CaseStatus, SupportCaseState, utc_now
from rag.resource_store import ResourceStore, VerifiedResourceStore
from state.case_store import CaseStore


ToolHandler = Callable[[AgentRequest, dict[str, Any]], Any | Awaitable[Any]]


class ToolError(RuntimeError):
    pass


class ToolNotFoundError(ToolError):
    pass


class ToolPermissionError(ToolError):
    pass


class ToolArgumentError(ToolError):
    pass


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: ToolHandler

    def as_llm_tool(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, tool: ToolSpec) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool 已注册: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> ToolSpec:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise ToolNotFoundError(f"未知 Tool: {name}") from exc

    def llm_tools(self, allowed_tools: set[str]) -> list[dict[str, Any]]:
        return [
            self._tools[name].as_llm_tool()
            for name in sorted(allowed_tools)
            if name in self._tools
        ]

    async def execute(
        self,
        *,
        tool_name: str,
        arguments: dict[str, Any],
        request: AgentRequest,
        allowed_tools: set[str],
    ) -> Any:
        if tool_name not in allowed_tools:
            raise ToolPermissionError(f"当前 Agent 无权执行 Tool: {tool_name}")

        spec = self.get(tool_name)
        self._validate_arguments(spec, arguments)
        result = spec.handler(request, arguments)
        if inspect.isawaitable(result):
            return await result
        return result

    @staticmethod
    def _validate_arguments(spec: ToolSpec, arguments: dict[str, Any]) -> None:
        if not isinstance(arguments, dict):
            raise ToolArgumentError("Tool arguments 必须是 object")

        schema = spec.input_schema
        required = schema.get("required", [])
        missing = [name for name in required if name not in arguments]
        if missing:
            raise ToolArgumentError(f"缺少必填参数: {', '.join(missing)}")

        if schema.get("additionalProperties") is False:
            allowed = set(schema.get("properties", {}).keys())
            extras = sorted(set(arguments) - allowed)
            if extras:
                raise ToolArgumentError(f"出现未声明参数: {', '.join(extras)}")

        properties = schema.get("properties", {})
        for name, value in arguments.items():
            rule = properties.get(name, {})
            expected = rule.get("type")
            if expected and not _matches_json_type(value, expected):
                raise ToolArgumentError(
                    f"参数 {name} 类型错误，期望 {expected}，实际 {type(value).__name__}"
                )
            enum_values = rule.get("enum")
            if enum_values and value not in enum_values:
                raise ToolArgumentError(
                    f"参数 {name} 取值错误，允许: {', '.join(map(str, enum_values))}"
                )


def _matches_json_type(value: Any, expected: str) -> bool:
    mapping = {
        "string": lambda x: isinstance(x, str),
        "integer": lambda x: isinstance(x, int) and not isinstance(x, bool),
        "number": lambda x: isinstance(x, (int, float)) and not isinstance(x, bool),
        "boolean": lambda x: isinstance(x, bool),
        "object": lambda x: isinstance(x, dict),
        "array": lambda x: isinstance(x, list),
    }
    checker = mapping.get(expected)
    return True if checker is None else checker(value)


def make_tool(
    *,
    name: str,
    description: str,
    properties: dict[str, Any],
    handler: ToolHandler,
    required: list[str] | None = None,
) -> ToolSpec:
    return ToolSpec(
        name=name,
        description=description,
        input_schema={
            "type": "object",
            "properties": properties,
            "required": required or [],
            "additionalProperties": False,
        },
        handler=handler,
    )


def _dedupe_text(items: list[str], *, max_items: int = 30) -> list[str]:
    result: list[str] = []
    for raw in items:
        value = str(raw).strip()
        if value and value not in result:
            result.append(value[:300])
        if len(result) >= max_items:
            break
    return result


def inspect_request_context(
    request: AgentRequest,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    """返回当前请求的结构化摘要，不访问外部系统。"""
    return {
        "focus": str(arguments.get("focus", "general"))[:60],
        "intent": request.intent_result.intent.value,
        "risk_level": request.risk_assessment.risk_level.value,
        "risk_signals": request.risk_assessment.signals,
        "autonomy_level": request.policy.autonomy_level,
        "case_status": request.case_state.status.value if request.case_state else None,
    }


def build_default_tool_registry(
    *,
    case_store: CaseStore,
    resource_store: ResourceStore | None = None,
) -> ToolRegistry:
    registry = ToolRegistry()
    resource_store = resource_store or VerifiedResourceStore.demo()

    registry.register(
        make_tool(
            name="inspect_request_context",
            description="读取当前请求已经完成的 Intent、Risk、Policy 与 CaseState 摘要。",
            properties={
                "focus": {
                    "type": "string",
                    "description": "希望查看的上下文重点，例如 risk、intent 或 case。",
                }
            },
            required=[],
            handler=inspect_request_context,
        )
    )

    def search_verified_resources(
        request: AgentRequest,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        query = str(arguments.get("query") or request.message).strip()
        top_k = int(arguments.get("top_k", 3))
        hits = resource_store.search(query, top_k=top_k)
        return {
            "query": query,
            "found": bool(hits),
            "count": len(hits),
            "evidence": [hit.to_dict() for hit in hits],
            "grounding_rule": (
                "只能依据 evidence 中的内容陈述可核验校园事实；"
                "如果 found=false 或证据没有覆盖具体地址、电话、时间、预约规则，必须明确无法确认。"
            ),
        }

    registry.register(
        make_tool(
            name="search_verified_resources",
            description=(
                "从 CampusMind 当前已验证资源库检索校园支持、就业或高风险支持边界信息。"
                "具体地址、电话、开放时间、预约规则等事实必须以返回 evidence 为依据。"
            ),
            properties={
                "query": {
                    "type": "string",
                    "description": "需要检索的资源问题或用户原始需求。",
                },
                "top_k": {
                    "type": "integer",
                    "description": "最多返回多少条证据，建议 1-5。",
                },
            },
            required=["query"],
            handler=search_verified_resources,
        )
    )

    def get_support_case(
        request: AgentRequest,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        state = case_store.get(
            user_id=request.user_id,
            conversation_id=request.conversation_id,
        )
        if state is None:
            return {"found": False, "case": None}
        return {"found": True, "case": state.model_dump(mode="json")}

    registry.register(
        make_tool(
            name="get_support_case",
            description="读取当前会话对应的结构化 SupportCaseState。",
            properties={},
            required=[],
            handler=get_support_case,
        )
    )

    def upsert_support_case(
        request: AgentRequest,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        state = case_store.get(
            user_id=request.user_id,
            conversation_id=request.conversation_id,
        )
        if state is None:
            state = SupportCaseState(
                case_id=f"case:{request.user_id}:{request.conversation_id}",
                user_id=request.user_id,
                conversation_id=request.conversation_id,
                topic=request.intent_result.intent.value,
                current_risk_level=request.risk_assessment.risk_level,
            )

        goal = arguments.get("goal")
        if isinstance(goal, str) and goal.strip():
            state.goal = goal.strip()[:500]

        if "add_known_context" in arguments:
            state.known_context = _dedupe_text(
                state.known_context + list(arguments.get("add_known_context") or [])
            )
        if "add_actions_done" in arguments:
            state.actions_done = _dedupe_text(
                state.actions_done + list(arguments.get("add_actions_done") or [])
            )
        if "set_next_actions" in arguments:
            state.next_actions = _dedupe_text(
                list(arguments.get("set_next_actions") or [])
            )
        if "add_resource_refs" in arguments:
            state.resource_refs = _dedupe_text(
                state.resource_refs + list(arguments.get("add_resource_refs") or [])
            )
        if "status" in arguments:
            state.status = CaseStatus(arguments["status"])

        # 风险等级不能由 LLM 参数覆盖，只能使用 RiskAnalyzer 的当前结果。
        state.current_risk_level = request.risk_assessment.risk_level
        state.topic = request.intent_result.intent.value
        state.updated_at = utc_now()
        saved = case_store.save(state)
        return {"updated": True, "case": saved.model_dump(mode="json")}

    registry.register(
        make_tool(
            name="upsert_support_case",
            description=(
                "更新当前 SupportCaseState 的目标、已完成动作、下一步、上下文或资源引用。"
                "不能修改 risk_level；风险等级由 RiskAnalyzer 决定。"
            ),
            properties={
                "goal": {"type": "string"},
                "add_known_context": {"type": "array"},
                "add_actions_done": {"type": "array"},
                "set_next_actions": {"type": "array"},
                "add_resource_refs": {"type": "array"},
                "status": {
                    "type": "string",
                    "enum": [status.value for status in CaseStatus],
                },
            },
            required=[],
            handler=upsert_support_case,
        )
    )

    return registry
