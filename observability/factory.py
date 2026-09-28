"""Trace backend 工厂。"""
from __future__ import annotations

from core.config import Settings
from observability.trace_store import InMemoryTraceStore, RedisTraceStore, TraceStore


def build_trace_store(settings: Settings) -> TraceStore:
    backend = settings.trace_backend
    if backend == "in_memory":
        return InMemoryTraceStore(max_items=min(settings.trace_max_items, 5000))
    if backend == "redis":
        return RedisTraceStore.from_url(
            settings.redis_url,
            ttl_seconds=settings.trace_ttl_seconds,
            max_items=settings.trace_max_items,
        )
    raise ValueError("TRACE_BACKEND 只支持 in_memory 或 redis")
