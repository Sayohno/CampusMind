"""CampusMind 轻量 RAG 资源检索层。

当前版本使用本地 JSON 演示知识库 + 确定性关键词打分，目的不是追求检索算法复杂度，
而是先把“外部证据 -> Tool -> LLM”的工程链路跑通。
后续可以替换为 Chroma / pgvector，而不改变 Tool 名称和上层 Agent Runtime。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ResourceDocument:
    resource_id: str
    title: str
    content: str
    keywords: tuple[str, ...]
    source_ref: str
    verified: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ResourceDocument":
        return cls(
            resource_id=str(data["resource_id"]),
            title=str(data["title"]),
            content=str(data["content"]),
            keywords=tuple(str(item) for item in data.get("keywords", [])),
            source_ref=str(data["source_ref"]),
            verified=bool(data.get("verified", False)),
        )


@dataclass(frozen=True)
class ResourceSearchHit:
    resource_id: str
    title: str
    content: str
    source_ref: str
    score: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "resource_id": self.resource_id,
            "title": self.title,
            "content": self.content,
            "source_ref": self.source_ref,
            "score": self.score,
        }


class VerifiedResourceStore:
    """只返回标记为 verified 的本地资源。"""

    def __init__(self, documents: list[ResourceDocument]) -> None:
        self._documents = [doc for doc in documents if doc.verified]

    @classmethod
    def from_json(cls, path: str | Path) -> "VerifiedResourceStore":
        file_path = Path(path)
        data = json.loads(file_path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("资源知识库根节点必须是 list")
        return cls([ResourceDocument.from_dict(item) for item in data])

    @classmethod
    def demo(cls) -> "VerifiedResourceStore":
        path = Path(__file__).resolve().parent / "corpus" / "demo_resources.json"
        return cls.from_json(path)

    def search(self, query: str, *, top_k: int = 3) -> list[ResourceSearchHit]:
        query = query.strip().lower()
        if not query:
            return []
        top_k = max(1, min(int(top_k), 10))

        scored: list[ResourceSearchHit] = []
        for doc in self._documents:
            score = self._score(query, doc)
            if score <= 0:
                continue
            scored.append(
                ResourceSearchHit(
                    resource_id=doc.resource_id,
                    title=doc.title,
                    content=doc.content,
                    source_ref=doc.source_ref,
                    score=score,
                )
            )

        scored.sort(key=lambda item: (-item.score, item.resource_id))
        return scored[:top_k]

    @staticmethod
    def _score(query: str, doc: ResourceDocument) -> int:
        """简单、可测试的关键词相关性打分。

        关键词命中权重高于标题/正文的整句命中。这个实现是第一版 Retrieval，
        重点是稳定和可解释，后续再替换为 embedding / rerank。
        """
        score = 0
        title = doc.title.lower()
        content = doc.content.lower()
        for keyword in doc.keywords:
            key = keyword.strip().lower()
            if key and key in query:
                score += 5

        # 对很短但直接出现的查询片段再给一点权重。
        compact_query = query.replace("？", "").replace("?", "").strip()
        if 2 <= len(compact_query) <= 20:
            if compact_query in title:
                score += 3
            if compact_query in content:
                score += 2
        return score
