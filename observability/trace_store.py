"""CampusMind 请求 Trace 存储。

提供进程内与 Redis 两种实现。
Trace 只保存已经脱敏后的 TraceRecord，
不保存原始用户消息和 Tool 参数值。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
from threading import Lock

from core.schemas import TraceRecord


class TraceStore(ABC):
    @abstractmethod
    def save(
        self,
        trace: TraceRecord,
    ) -> TraceRecord:
        raise NotImplementedError

    @abstractmethod
    def get(
        self,
        request_id: str,
    ) -> TraceRecord | None:
        raise NotImplementedError

    @abstractmethod
    def recent(
        self,
        limit: int = 20,
    ) -> list[TraceRecord]:
        raise NotImplementedError


class InMemoryTraceStore(TraceStore):
    """有界、线程安全的进程内 Trace Store。"""

    def __init__(
        self,
        max_items: int = 500,
    ) -> None:
        if max_items < 1:
            raise ValueError(
                "max_items 必须 >= 1"
            )

        self.max_items = max_items
        self._items: OrderedDict[
            str,
            TraceRecord,
        ] = OrderedDict()
        self._lock = Lock()

    def save(
        self,
        trace: TraceRecord,
    ) -> TraceRecord:
        with self._lock:
            self._items[
                trace.request_id
            ] = trace

            self._items.move_to_end(
                trace.request_id
            )

            while (
                len(self._items)
                > self.max_items
            ):
                self._items.popitem(
                    last=False
                )

        return trace

    def get(
        self,
        request_id: str,
    ) -> TraceRecord | None:
        with self._lock:
            item = self._items.get(
                request_id
            )

            return (
                item.model_copy(deep=True)
                if item is not None
                else None
            )

    def recent(
        self,
        limit: int = 20,
    ) -> list[TraceRecord]:
        limit = max(
            1,
            min(limit, 100),
        )

        with self._lock:
            values = list(
                self._items.values()
            )[-limit:]

            return [
                item.model_copy(deep=True)
                for item in reversed(values)
            ]


class RedisTraceStore(TraceStore):
    """Redis String + Sorted Set Trace Store。

    单条 Trace 使用 TTL 自动清理。
    Sorted Set 保存 request_id 与时间分数，
    用于 recent 查询。

    索引同时限制 max_items，
    防止无限增长。
    """

    def __init__(
        self,
        client,
        *,
        ttl_seconds: int = 7 * 24 * 60 * 60,
        max_items: int = 2000,
        key_prefix: str = "campusmind:trace",
    ) -> None:
        if ttl_seconds < 0:
            raise ValueError(
                "ttl_seconds 不能小于 0"
            )

        if max_items < 1:
            raise ValueError(
                "max_items 必须 >= 1"
            )

        self.client = client
        self.ttl_seconds = ttl_seconds
        self.max_items = max_items
        self.key_prefix = key_prefix.rstrip(
            ":"
        )
        self.index_key = (
            f"{self.key_prefix}:index"
        )

    @classmethod
    def from_url(
        cls,
        url: str,
        **kwargs,
    ) -> "RedisTraceStore":
        try:
            import redis
        except ImportError as exc:
            raise RuntimeError(
                "Redis Trace 后端需要安装 "
                "redis Python 包"
            ) from exc

        client = redis.Redis.from_url(
            url,
            decode_responses=True,
        )

        return cls(
            client,
            **kwargs,
        )

    def _key(
        self,
        request_id: str,
    ) -> str:
        return (
            f"{self.key_prefix}:item:"
            f"{request_id}"
        )

    def save(
        self,
        trace: TraceRecord,
    ) -> TraceRecord:
        key = self._key(
            trace.request_id
        )

        payload = (
            trace.model_dump_json()
        )

        if self.ttl_seconds > 0:
            self.client.set(
                key,
                payload,
                ex=self.ttl_seconds,
            )
        else:
            self.client.set(
                key,
                payload,
            )

        self.client.zadd(
            self.index_key,
            {
                trace.request_id:
                    trace.created_at.timestamp()
            },
        )

        size = int(
            self.client.zcard(
                self.index_key
            )
        )

        overflow = (
            size - self.max_items
        )

        if overflow > 0:
            stale_ids = (
                self.client.zrange(
                    self.index_key,
                    0,
                    overflow - 1,
                )
            )

            if stale_ids:
                self.client.delete(
                    *[
                        self._key(item)
                        for item
                        in stale_ids
                    ]
                )

                self.client.zrem(
                    self.index_key,
                    *stale_ids,
                )

        return trace

    def get(
        self,
        request_id: str,
    ) -> TraceRecord | None:
        raw = self.client.get(
            self._key(request_id)
        )

        if raw is None:
            self.client.zrem(
                self.index_key,
                request_id,
            )
            return None

        return (
            TraceRecord
            .model_validate_json(raw)
        )

    def recent(
        self,
        limit: int = 20,
    ) -> list[TraceRecord]:
        limit = max(
            1,
            min(limit, 100),
        )

        candidate_ids = (
            self.client.zrevrange(
                self.index_key,
                0,
                max(
                    limit * 3 - 1,
                    limit - 1,
                ),
            )
        )

        result: list[
            TraceRecord
        ] = []

        stale: list[str] = []

        for request_id in candidate_ids:
            raw = self.client.get(
                self._key(request_id)
            )

            if raw is None:
                stale.append(
                    request_id
                )
                continue

            result.append(
                TraceRecord
                .model_validate_json(raw)
            )

            if len(result) >= limit:
                break

        if stale:
            self.client.zrem(
                self.index_key,
                *stale,
            )

        return result