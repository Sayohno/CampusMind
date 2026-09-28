"""Conversation Memory 抽象、进程内实现与 Redis 实现。"""
from __future__ import annotations

from abc import ABC, abstractmethod
from urllib.parse import quote

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


class RedisConversationMemory(ConversationMemory):
    """Redis List 实现的短期 Conversation Memory。

    - 每个 conversation 对应一个 List；
    - RPUSH 追加消息，LTRIM 保留最近 N 条；
    - 每次写入刷新 TTL，因此活跃会话会续期。
    """

    def __init__(
        self,
        client,
        *,
        ttl_seconds: int = 7 * 24 * 60 * 60,
        max_messages_per_conversation: int = 50,
        key_prefix: str = "campusmind:memory",
    ) -> None:
        if ttl_seconds < 0:
            raise ValueError("ttl_seconds 不能小于 0")
        if max_messages_per_conversation < 2:
            raise ValueError("max_messages_per_conversation 必须 >= 2")
        self.client = client
        self.ttl_seconds = ttl_seconds
        self.max_messages_per_conversation = max_messages_per_conversation
        self.key_prefix = key_prefix.rstrip(":")

    @classmethod
    def from_url(cls, url: str, **kwargs) -> "RedisConversationMemory":
        try:
            import redis
        except ImportError as exc:  # pragma: no cover - 仅在缺依赖时触发
            raise RuntimeError("Redis 后端需要安装 redis Python 包") from exc
        client = redis.Redis.from_url(url, decode_responses=True)
        return cls(client, **kwargs)

    def _key(self, user_id: str, conversation_id: str) -> str:
        return (
            f"{self.key_prefix}:"
            f"{quote(user_id, safe='')}:{quote(conversation_id, safe='')}"
        )

    def get_recent(
        self,
        *,
        user_id: str,
        conversation_id: str,
        limit: int = 8,
    ) -> list[ConversationMessage]:
        if limit < 0:
            raise ValueError("limit 不能小于 0")
        if limit == 0:
            return []
        key = self._key(user_id, conversation_id)
        raw_items = self.client.lrange(key, -limit, -1)
        return [ConversationMessage.model_validate_json(item) for item in raw_items]

    def append(
        self,
        *,
        user_id: str,
        conversation_id: str,
        message: ConversationMessage,
    ) -> None:
        key = self._key(user_id, conversation_id)
        self.client.rpush(key, message.model_dump_json())
        self.client.ltrim(key, -self.max_messages_per_conversation, -1)
        if self.ttl_seconds > 0:
            self.client.expire(key, self.ttl_seconds)

    def count(self, *, user_id: str, conversation_id: str) -> int:
        return int(self.client.llen(self._key(user_id, conversation_id)))
