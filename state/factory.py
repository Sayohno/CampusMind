"""状态后端工厂：把 Orchestrator 与具体存储实现解耦。"""
from __future__ import annotations

from core.config import Settings
from state.case_store import CaseStore, InMemoryCaseStore, RedisCaseStore
from state.memory_store import (
    ConversationMemory,
    InMemoryConversationMemory,
    RedisConversationMemory,
)


def build_state_stores(settings: Settings) -> tuple[ConversationMemory, CaseStore]:
    backend = settings.state_backend
    if backend == "in_memory":
        return InMemoryConversationMemory(), InMemoryCaseStore()
    if backend == "redis":
        memory = RedisConversationMemory.from_url(
            settings.redis_url,
            ttl_seconds=settings.memory_ttl_seconds,
        )
        cases = RedisCaseStore.from_url(
            settings.redis_url,
            ttl_seconds=settings.case_ttl_seconds,
        )
        return memory, cases
    raise ValueError("STATE_BACKEND 只支持 in_memory 或 redis")
