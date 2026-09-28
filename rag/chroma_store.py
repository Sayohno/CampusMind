"""ChromaDB Vector RAG backend。

Chroma 只负责向量索引/检索；Grounding、Abstention 与安全 Policy 仍由上层 Runtime / Guard 控制。
"""
from __future__ import annotations

from pathlib import Path

from rag.embeddings import DeterministicHashEmbedder, EmbeddingModel
from rag.resource_store import (
    ResourceDocument,
    ResourceSearchHit,
    ResourceStore,
    demo_documents,
)


class ChromaResourceStore(ResourceStore):
    def __init__(
        self,
        documents: list[ResourceDocument],
        *,
        persist_directory: str,
        collection_name: str = "campusmind_resources",
        embedder: EmbeddingModel | None = None,
        min_score: float = 0.12,
        client=None,
    ) -> None:
        self._documents = [doc for doc in documents if doc.verified]
        self._by_id = {doc.resource_id: doc for doc in self._documents}
        self.embedder = embedder or DeterministicHashEmbedder()
        self.min_score = float(min_score)

        if client is None:
            try:
                import chromadb
            except ImportError as exc:  # pragma: no cover - Docker 安装依赖后不触发
                raise RuntimeError("Chroma RAG 后端需要安装 chromadb Python 包") from exc
            Path(persist_directory).mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(path=persist_directory)
        self.client = client
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._sync_documents()

    @classmethod
    def demo(
        cls,
        *,
        persist_directory: str,
        collection_name: str = "campusmind_resources",
        dimensions: int = 384,
        min_score: float = 0.12,
    ) -> "ChromaResourceStore":
        return cls(
            demo_documents(),
            persist_directory=persist_directory,
            collection_name=collection_name,
            embedder=DeterministicHashEmbedder(dimensions),
            min_score=min_score,
        )

    def _sync_documents(self) -> None:
        target_ids = [doc.resource_id for doc in self._documents]
        existing = self.collection.get()
        existing_ids = set(existing.get("ids") or [])
        stale_ids = sorted(existing_ids - set(target_ids))
        if stale_ids:
            self.collection.delete(ids=stale_ids)

        if not self._documents:
            return
        embeddings = self.embedder.embed([doc.embedding_text() for doc in self._documents])
        self.collection.upsert(
            ids=target_ids,
            embeddings=embeddings,
            documents=[doc.content for doc in self._documents],
            metadatas=[
                {
                    "resource_id": doc.resource_id,
                    "title": doc.title,
                    "source_ref": doc.source_ref,
                    "verified": True,
                }
                for doc in self._documents
            ],
        )

    def search(self, query: str, *, top_k: int = 3) -> list[ResourceSearchHit]:
        query = query.strip()
        if not query or not self._documents:
            return []
        top_k = max(1, min(int(top_k), 10, len(self._documents)))
        query_vector = self.embedder.embed([query])[0]
        result = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        ids = (result.get("ids") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        hits: list[ResourceSearchHit] = []
        for index, resource_id in enumerate(ids):
            distance = float(distances[index]) if index < len(distances) else 1.0
            score = max(0.0, min(1.0, 1.0 - distance))
            if score < self.min_score:
                continue
            metadata = metadatas[index] if index < len(metadatas) and metadatas[index] else {}
            fallback = self._by_id.get(str(resource_id))
            content = documents[index] if index < len(documents) else None
            hits.append(
                ResourceSearchHit(
                    resource_id=str(resource_id),
                    title=str(metadata.get("title") or (fallback.title if fallback else "")),
                    content=str(content or (fallback.content if fallback else "")),
                    source_ref=str(
                        metadata.get("source_ref") or (fallback.source_ref if fallback else "")
                    ),
                    score=score,
                )
            )
        return hits
