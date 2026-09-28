"""CampusMind LLM 适配层。

上层 Agent Runtime 只认识统一的 LLMResponse，不直接依赖 OpenAI、DeepSeek
等厂商原始响应格式。
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections import deque
from typing import Any, Iterable

import httpx

from core.schemas import LLMResponse, ToolCall


class LLMClientError(RuntimeError):
    """模型调用或响应解析失败。"""


class BaseLLMClient(ABC):
    @abstractmethod
    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        raise NotImplementedError


class OpenAICompatibleLLMClient(BaseLLMClient):
    """OpenAI-compatible Chat Completions 适配器。

    可用于支持 /chat/completions 兼容协议的供应商。具体 base_url、model、API Key
    从环境配置传入，不把厂商信息写死在 Agent Runtime 中。
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        timeout: float = 60.0,
        temperature: float = 0.2,
    ) -> None:
        if not api_key:
            raise ValueError("api_key 不能为空")
        if not base_url:
            raise ValueError("base_url 不能为空")
        if not model:
            raise ValueError("model 不能为空")

        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.temperature = temperature

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        url = f"{self.base_url}/chat/completions"

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(url, headers=headers, json=payload)
                response.raise_for_status()
        except httpx.HTTPError as exc:
            raise LLMClientError(f"LLM HTTP 调用失败: {exc}") from exc

        try:
            data = response.json()
            message = data["choices"][0]["message"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise LLMClientError("LLM 返回格式不符合 Chat Completions 预期") from exc

        parsed_calls: list[ToolCall] = []
        for raw_call in message.get("tool_calls") or []:
            try:
                function = raw_call["function"]
                raw_arguments = function.get("arguments", "{}") or "{}"
                arguments = (
                    raw_arguments
                    if isinstance(raw_arguments, dict)
                    else json.loads(raw_arguments)
                )
                if not isinstance(arguments, dict):
                    raise TypeError("tool arguments 必须是 object")
                parsed_calls.append(
                    ToolCall(
                        id=str(raw_call.get("id") or f"tool_{len(parsed_calls) + 1}"),
                        name=str(function["name"]),
                        arguments=arguments,
                    )
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise LLMClientError("LLM tool_call 参数解析失败") from exc

        return LLMResponse(
            content=message.get("content"),
            tool_calls=parsed_calls,
        )


class StaticLLMClient(BaseLLMClient):
    """离线演示 / 路由测试用，不访问真实模型。"""

    def __init__(self, content: str = "CampusMind Runtime 已收到请求。") -> None:
        self.content = content
        self.call_count = 0

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        self.call_count += 1
        return LLMResponse(content=self.content)


class ScriptedLLMClient(BaseLLMClient):
    """确定性 Runtime 单元测试用：按预设顺序返回 LLMResponse。"""

    def __init__(self, responses: Iterable[LLMResponse]) -> None:
        self._responses = deque(responses)
        self.call_count = 0
        self.seen_messages: list[list[dict[str, Any]]] = []
        self.seen_tools: list[list[dict[str, Any]]] = []

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        self.call_count += 1
        self.seen_messages.append([dict(item) for item in messages])
        self.seen_tools.append(list(tools or []))
        if not self._responses:
            raise LLMClientError("ScriptedLLMClient 没有更多预设响应")
        return self._responses.popleft()
