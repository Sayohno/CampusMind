"""CampusMind 请求 Trace 存储。

当前版本使用进程内存储，便于先验证可观测性契约。后续可替换 Redis / 数据库，
API 与 Orchestrator 不需要依赖具体存储实现。
"""
from __future__ import annotations

from collections import OrderedDict
from threading import Lock

from core.schemas import TraceRecord


class InMemoryTraceStore:
    """有界、线程安全的进程内 Trace Store。"""

    def __init__(self, max_items: int = 500) -> None:
        if max_items < 1:
            raise ValueError("max_items 必须 >= 1")
        self.max_items = max_items
        self._items: OrderedDict[str, TraceRecord] = OrderedDict()
        self._lock = Lock()

    def save(self, trace: TraceRecord) -> TraceRecord:
        with self._lock:
            self._items[trace.request_id] = trace
            self._items.move_to_end(trace.request_id)
            while len(self._items) > self.max_items:
                self._items.popitem(last=False)
        return trace

    def get(self, request_id: str) -> TraceRecord | None:
        with self._lock:
            item = self._items.get(request_id)
            return item.model_copy(deep=True) if item is not None else None

    def recent(self, limit: int = 20) -> list[TraceRecord]:
        limit = max(1, min(limit, 100))
        with self._lock:
            values = list(self._items.values())[-limit:]
            return [item.model_copy(deep=True) for item in reversed(values)]
