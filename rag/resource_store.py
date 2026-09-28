"""CampusMind Verified Resource RAG 存储抽象。

V1.1 保留确定性关键词基线，同时增加纯 Python VectorResourceStore；Chroma 后端在
rag/chroma_store.py 中实现。上层 Tool 只依赖 ResourceStore.search()，因此检索后端可替换。
"""
from __future__ import annotations

import json
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rag.embeddings import DeterministicHashEmbedder, EmbeddingModel


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

    def embedding_text(self) -> str:
        keywords = " ".join(self.keywords)
        # 轻量离线向量器没有预训练语义知识，因此对人工维护的 verified keywords 做显式加权。
        keyword_boost = " ".join([keywords] * 4)
        return f"{self.title}\n关键词：{keyword_boost}\n{self.content}"


@dataclass(frozen=True)
class ResourceSearchHit:
    resource_id: str
    title: str
    content: str
    source_ref: str
    score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "resource_id": self.resource_id,
            "title": self.title,
            "content": self.content,
            "source_ref": self.source_ref,
            "score": round(float(self.score), 4),
        }


class ResourceStore(ABC):
    @abstractmethod
    def search(self, query: str, *, top_k: int = 3) -> list[ResourceSearchHit]:
        raise NotImplementedError


def load_documents(path: str | Path) -> list[ResourceDocument]:
    file_path = Path(path)
    data = json.loads(file_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("资源知识库根节点必须是 list")
    return [ResourceDocument.from_dict(item) for item in data]


def demo_documents() -> list[ResourceDocument]:
    path = Path(__file__).resolve().parent / "corpus" / "demo_resources.json"
    return load_documents(path)


class VerifiedResourceStore(ResourceStore):
    """关键词 Retrieval 基线：只返回标记为 verified 的本地资源。"""

    def __init__(self, documents: list[ResourceDocument]) -> None:
        self._documents = [doc for doc in documents if doc.verified]

    @classmethod
    def from_json(cls, path: str | Path) -> "VerifiedResourceStore":
        return cls(load_documents(path))

    @classmethod
    def demo(cls) -> "VerifiedResourceStore":
        return cls(demo_documents())

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
                    score=float(score),
                )
            )

        scored.sort(key=lambda item: (-item.score, item.resource_id))
        return scored[:top_k]

    @staticmethod
    def _score(query: str, doc: ResourceDocument) -> int:
        score = 0
        title = doc.title.lower()
        content = doc.content.lower()
        for keyword in doc.keywords:
            key = keyword.strip().lower()
            if key and key in query:
                score += 5

        compact_query = query.replace("？", "").replace("?", "").strip()
        if 2 <= len(compact_query) <= 20:
            if compact_query in title:
                score += 3
            if compact_query in content:
                score += 2
        return score


class VectorResourceStore(ResourceStore):
    """纯 Python 内存 Vector Retrieval，用于离线测试和 Chroma 行为基线。"""

    def __init__(
        self,
        documents: list[ResourceDocument],
        *,
        embedder: EmbeddingModel | None = None,
        min_score: float = 0.12,
    ) -> None:
        self._documents = [doc for doc in documents if doc.verified]
        self.embedder = embedder or DeterministicHashEmbedder()
        self.min_score = float(min_score)
        self._vectors = self.embedder.embed([doc.embedding_text() for doc in self._documents])

    @classmethod
    def demo(
        cls,
        *,
        dimensions: int = 384,
        min_score: float = 0.12,
    ) -> "VectorResourceStore":
        return cls(
            demo_documents(),
            embedder=DeterministicHashEmbedder(dimensions),
            min_score=min_score,
        )

    def search(self, query: str, *, top_k: int = 3) -> list[ResourceSearchHit]:
        query = query.strip()
        if not query or not self._documents:
            return []
        top_k = max(1, min(int(top_k), 10))
        query_vector = self.embedder.embed([query])[0]

        hits: list[ResourceSearchHit] = []
        for doc, vector in zip(self._documents, self._vectors):
            score = self._cosine(query_vector, vector)
            if score < self.min_score:
                continue
            hits.append(
                ResourceSearchHit(
                    resource_id=doc.resource_id,
                    title=doc.title,
                    content=doc.content,
                    source_ref=doc.source_ref,
                    score=score,
                )
            )
        hits.sort(key=lambda item: (-item.score, item.resource_id))
        return hits[:top_k]

    @staticmethod
    def _cosine(left: list[float], right: list[float]) -> float:
        # Embedder 已经 L2 normalize，因此 dot product 即 cosine similarity。
        value = sum(a * b for a, b in zip(left, right))
        return max(0.0, min(1.0, float(value)))
