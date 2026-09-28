"""Conversation Memory 抽象与进程内实现。

当前先用 InMemoryConversationMemory 验证读写链路；接口保持独立，后续可替换为 Redis，
上层 Orchestrator / Agent 不需要改业务逻辑。
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from core.schemas import ConversationMessage


class ConversationMemory(ABC):
    @abstractmethod
    def get_recent(
        self,
        *,
        user_id: str,
        conversation_id: str,
        limit: int = 8,
    ) -> list[ConversationMessage]:
        raise NotImplementedError

    @abstractmethod
    def append(
        self,
        *,
        user_id: str,
        conversation_id: str,
        message: ConversationMessage,
    ) -> None:
        raise NotImplementedError

    def append_turn(
        self,
        *,
        user_id: str,
        conversation_id: str,
        user_message: str,
        assistant_message: str,
    ) -> None:
        self.append(
            user_id=user_id,
            conversation_id=conversation_id,
            message=ConversationMessage(role="user", content=user_message),
        )
        self.append(
            user_id=user_id,
            conversation_id=conversation_id,
            message=ConversationMessage(role="assistant", content=assistant_message),
        )

    @abstractmethod
    def count(self, *, user_id: str, conversation_id: str) -> int:
        raise NotImplementedError


class InMemoryConversationMemory(ConversationMemory):
    def __init__(self, *, max_messages_per_conversation: int = 50) -> None:
        if max_messages_per_conversation < 2:
            raise ValueError("max_messages_per_conversation 必须 >= 2")
        self.max_messages_per_conversation = max_messages_per_conversation
        self._messages: dict[tuple[str, str], list[ConversationMessage]] = {}

    def get_recent(
        self,
        *,
        user_id: str,
        conversation_id: str,
        limit: int = 8,
    ) -> list[ConversationMessage]:
        if limit < 0:
            raise ValueError("limit 不能小于 0")
        key = (user_id, conversation_id)
        items = self._messages.get(key, [])
        if limit == 0:
            return []
        return [item.model_copy(deep=True) for item in items[-limit:]]

    def append(
        self,
        *,
        user_id: str,
        conversation_id: str,
        message: ConversationMessage,
    ) -> None:
        key = (user_id, conversation_id)
        bucket = self._messages.setdefault(key, [])
        bucket.append(message.model_copy(deep=True))
        if len(bucket) > self.max_messages_per_conversation:
            del bucket[: len(bucket) - self.max_messages_per_conversation]

    def count(self, *, user_id: str, conversation_id: str) -> int:
        return len(self._messages.get((user_id, conversation_id), []))
