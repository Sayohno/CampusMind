import unittest

from core.schemas import ConversationMessage, RiskLevel, SupportCaseState, TraceRecord
from observability.trace_store import RedisTraceStore
from state.case_store import RedisCaseStore
from state.memory_store import RedisConversationMemory


class FakeRedis:
    """只实现本测试需要的 Redis 命令，验证 key/TTL/序列化语义。"""

    def __init__(self):
        self.strings = {}
        self.lists = {}
        self.zsets = {}
        self.expirations = {}

    def rpush(self, key, value):
        self.lists.setdefault(key, []).append(value)
        return len(self.lists[key])

    def ltrim(self, key, start, stop):
        items = self.lists.get(key, [])
        if start < 0:
            start = max(len(items) + start, 0)
        if stop < 0:
            stop = len(items) + stop
        self.lists[key] = items[start : stop + 1]
        return True

    def lrange(self, key, start, stop):
        items = self.lists.get(key, [])
        if start < 0:
            start = max(len(items) + start, 0)
        if stop < 0:
            stop = len(items) + stop
        return items[start : stop + 1]

    def llen(self, key):
        return len(self.lists.get(key, []))

    def expire(self, key, seconds):
        self.expirations[key] = seconds
        return True

    def set(self, key, value, ex=None):
        self.strings[key] = value
        if ex is not None:
            self.expirations[key] = ex
        return True

    def get(self, key):
        return self.strings.get(key)

    def zadd(self, key, mapping):
        self.zsets.setdefault(key, {}).update(mapping)
        return len(mapping)

    def zcard(self, key):
        return len(self.zsets.get(key, {}))

    def zrange(self, key, start, stop):
        items = sorted(self.zsets.get(key, {}).items(), key=lambda item: (item[1], item[0]))
        return [name for name, _ in items[start : stop + 1]]

    def zrevrange(self, key, start, stop):
        items = sorted(
            self.zsets.get(key, {}).items(),
            key=lambda item: (item[1], item[0]),
            reverse=True,
        )
        return [name for name, _ in items[start : stop + 1]]

    def zrem(self, key, *members):
        bucket = self.zsets.get(key, {})
        removed = 0
        for member in members:
            if member in bucket:
                removed += 1
                bucket.pop(member)
        return removed

    def delete(self, *keys):
        removed = 0
        for key in keys:
            removed += int(key in self.strings or key in self.lists)
            self.strings.pop(key, None)
            self.lists.pop(key, None)
            self.expirations.pop(key, None)
        return removed


class RedisConversationMemoryTests(unittest.TestCase):
    def test_messages_persist_with_ttl_and_recent_limit(self):
        fake = FakeRedis()
        memory = RedisConversationMemory(
            fake,
            ttl_seconds=60,
            max_messages_per_conversation=3,
        )
        for content in ["m1", "m2", "m3", "m4"]:
            memory.append(
                user_id="u:1",
                conversation_id="c/1",
                message=ConversationMessage(role="user", content=content),
            )

        self.assertEqual(memory.count(user_id="u:1", conversation_id="c/1"), 3)
        recent = memory.get_recent(user_id="u:1", conversation_id="c/1", limit=2)
        self.assertEqual([item.content for item in recent], ["m3", "m4"])
        key = memory._key("u:1", "c/1")
        self.assertEqual(fake.expirations[key], 60)
        self.assertNotIn("/", key)


class RedisCaseStoreTests(unittest.TestCase):
    def test_case_round_trip_and_ttl(self):
        fake = FakeRedis()
        store = RedisCaseStore(fake, ttl_seconds=120)
        state = SupportCaseState(
            case_id="case:u:c",
            user_id="u",
            conversation_id="c",
            topic="academic_career",
            current_risk_level=RiskLevel.LOW,
            actions_done=["已完成简历初稿"],
        )
        saved = store.save(state)
        loaded = store.get(user_id="u", conversation_id="c")

        self.assertEqual(loaded.actions_done, ["已完成简历初稿"])
        self.assertEqual(loaded.case_id, saved.case_id)
        self.assertEqual(fake.expirations[store._key("u", "c")], 120)


class RedisTraceStoreTests(unittest.TestCase):
    def test_trace_round_trip_recent_and_max_items(self):
        fake = FakeRedis()
        store = RedisTraceStore(fake, ttl_seconds=30, max_items=2)
        for idx in range(3):
            store.save(
                TraceRecord(
                    request_id=f"req_{idx}",
                    status="success",
                    latency_ms=float(idx),
                )
            )

        self.assertIsNone(store.get("req_0"))
        recent = store.recent(limit=2)
        self.assertEqual({item.request_id for item in recent}, {"req_1", "req_2"})
        self.assertEqual(fake.expirations[store._key("req_2")], 30)


if __name__ == "__main__":
    unittest.main(verbosity=2)
