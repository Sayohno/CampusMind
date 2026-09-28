"""RAG backend 工厂。"""
from __future__ import annotations

from core.config import Settings
from rag.resource_store import ResourceStore, VectorResourceStore, VerifiedResourceStore


def build_resource_store(settings: Settings) -> ResourceStore:
    backend = settings.rag_backend
    if backend == "keyword":
        return VerifiedResourceStore.demo()
    if backend == "vector":
        return VectorResourceStore.demo(
            dimensions=settings.vector_dimensions,
            min_score=settings.vector_min_score,
        )
    if backend == "chroma":
        from rag.chroma_store import ChromaResourceStore

        return ChromaResourceStore.demo(
            persist_directory=settings.chroma_persist_dir,
            collection_name=settings.chroma_collection,
            dimensions=settings.vector_dimensions,
            min_score=settings.vector_min_score,
        )
    raise ValueError("RAG_BACKEND 只支持 keyword、vector 或 chroma")
