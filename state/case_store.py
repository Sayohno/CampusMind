"""SupportCaseState 持久化抽象、进程内实现与 Redis 实现。"""
from __future__ import annotations

from abc import ABC, abstractmethod
from urllib.parse import quote

from core.schemas import CaseStatus, RiskLevel, SupportCaseState, utc_now


class CaseStore(ABC):
    @abstractmethod
    def get(self, *, user_id: str, conversation_id: str) -> SupportCaseState | None:
        raise NotImplementedError

    @abstractmethod
    def save(self, state: SupportCaseState) -> SupportCaseState:
        raise NotImplementedError

    def ensure(
        self,
        *,
        user_id: str,
        conversation_id: str,
        topic: str,
        risk_level: RiskLevel,
    ) -> SupportCaseState:
        current = self.get(user_id=user_id, conversation_id=conversation_id)
        if current is None:
            current = SupportCaseState(
                case_id=f"case:{user_id}:{conversation_id}",
                user_id=user_id,
                conversation_id=conversation_id,
                topic=topic,
                current_risk_level=risk_level,
            )
        else:
            current.topic = topic
            current.current_risk_level = risk_level
            current.updated_at = utc_now()
        return self.save(current)

    def update_status(
        self,
        *,
        user_id: str,
        conversation_id: str,
        status: CaseStatus,
    ) -> SupportCaseState:
        state = self.get(user_id=user_id, conversation_id=conversation_id)
        if state is None:
            raise KeyError("SupportCaseState 不存在")
        state.status = status
        return self.save(state)


class InMemoryCaseStore(CaseStore):
    def __init__(self) -> None:
        self._states: dict[tuple[str, str], SupportCaseState] = {}

    def get(self, *, user_id: str, conversation_id: str) -> SupportCaseState | None:
        state = self._states.get((user_id, conversation_id))
        return None if state is None else state.model_copy(deep=True)

    def save(self, state: SupportCaseState) -> SupportCaseState:
        state.updated_at = utc_now()
        self._states[(state.user_id, state.conversation_id)] = state.model_copy(deep=True)
        return state.model_copy(deep=True)


class RedisCaseStore(CaseStore):
    """Redis String(JSON) 实现的 SupportCaseState。

    Demo 默认可设置较长 TTL。生产化时如果 CaseState 需要长期审计/事务能力，
    可以保持 CaseStore 接口不变，将实现替换为 PostgreSQL/MySQL。
    """

    def __init__(
        self,
        client,
        *,
        ttl_seconds: int = 30 * 24 * 60 * 60,
        key_prefix: str = "campusmind:case",
    ) -> None:
        if ttl_seconds < 0:
            raise ValueError("ttl_seconds 不能小于 0")
        self.client = client
        self.ttl_seconds = ttl_seconds
        self.key_prefix = key_prefix.rstrip(":")

    @classmethod
    def from_url(cls, url: str, **kwargs) -> "RedisCaseStore":
        try:
            import redis
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Redis 后端需要安装 redis Python 包") from exc
        client = redis.Redis.from_url(url, decode_responses=True)
        return cls(client, **kwargs)

    def _key(self, user_id: str, conversation_id: str) -> str:
        return (
            f"{self.key_prefix}:"
            f"{quote(user_id, safe='')}:{quote(conversation_id, safe='')}"
        )

    def get(self, *, user_id: str, conversation_id: str) -> SupportCaseState | None:
        raw = self.client.get(self._key(user_id, conversation_id))
        if raw is None:
            return None
        return SupportCaseState.model_validate_json(raw)

    def save(self, state: SupportCaseState) -> SupportCaseState:
        state = state.model_copy(deep=True)
        state.updated_at = utc_now()
        key = self._key(state.user_id, state.conversation_id)
        payload = state.model_dump_json()
        if self.ttl_seconds > 0:
            self.client.set(key, payload, ex=self.ttl_seconds)
        else:
            self.client.set(key, payload)
        return state.model_copy(deep=True)
