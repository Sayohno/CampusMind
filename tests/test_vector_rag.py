import math
import unittest

from rag.chroma_store import ChromaResourceStore
from rag.embeddings import DeterministicHashEmbedder
from rag.resource_store import ResourceDocument, VectorResourceStore, demo_documents


class FakeCollection:
    def __init__(self):
        self.ids = []
        self.embeddings = {}
        self.documents = {}
        self.metadatas = {}

    def get(self):
        return {"ids": list(self.ids)}

    def delete(self, ids):
        for item in ids:
            if item in self.ids:
                self.ids.remove(item)
            self.embeddings.pop(item, None)
            self.documents.pop(item, None)
            self.metadatas.pop(item, None)

    def upsert(self, *, ids, embeddings, documents, metadatas):
        for rid, vector, document, metadata in zip(ids, embeddings, documents, metadatas):
            if rid not in self.ids:
                self.ids.append(rid)
            self.embeddings[rid] = vector
            self.documents[rid] = document
            self.metadatas[rid] = metadata

    def query(self, *, query_embeddings, n_results, include):
        query = query_embeddings[0]
        scored = []
        for rid in self.ids:
            vector = self.embeddings[rid]
            cosine = sum(a * b for a, b in zip(query, vector))
            distance = 1.0 - cosine
            scored.append((distance, rid))
        scored.sort()
        selected = scored[:n_results]
        return {
            "ids": [[rid for _, rid in selected]],
            "documents": [[self.documents[rid] for _, rid in selected]],
            "metadatas": [[self.metadatas[rid] for _, rid in selected]],
            "distances": [[distance for distance, _ in selected]],
        }


class FakeChromaClient:
    def __init__(self):
        self.collection = FakeCollection()

    def get_or_create_collection(self, *, name, metadata):
        self.name = name
        self.metadata = metadata
        return self.collection


class HashEmbeddingTests(unittest.TestCase):
    def test_embedding_is_normalized_and_deterministic(self):
        embedder = DeterministicHashEmbedder(128)
        first = embedder.embed(["学校心理咨询预约"])[0]
        second = embedder.embed(["学校心理咨询预约"])[0]
        self.assertEqual(first, second)
        self.assertAlmostEqual(math.sqrt(sum(v * v for v in first)), 1.0, places=6)


class VectorResourceStoreTests(unittest.TestCase):
    def test_vector_retrieval_returns_counseling_evidence(self):
        store = VectorResourceStore.demo(dimensions=256, min_score=0.08)
        hits = store.search("我想预约学校的心理咨询服务", top_k=2)
        self.assertTrue(hits)
        self.assertEqual(hits[0].resource_id, "demo_counseling_access")

    def test_vector_retrieval_returns_no_unrelated_evidence_with_threshold(self):
        store = VectorResourceStore.demo(dimensions=256, min_score=0.20)
        hits = store.search("量子芯片晶圆光刻机天气预报", top_k=3)
        self.assertEqual(hits, [])

    def test_unverified_documents_are_not_indexed(self):
        docs = demo_documents() + [
            ResourceDocument(
                resource_id="unverified",
                title="未验证号码",
                content="这是不应进入检索的内容",
                keywords=("号码",),
                source_ref="demo://unverified",
                verified=False,
            )
        ]
        store = VectorResourceStore(docs, min_score=0.0)
        ids = {hit.resource_id for hit in store.search("号码", top_k=10)}
        self.assertNotIn("unverified", ids)


class ChromaAdapterTests(unittest.TestCase):
    def test_chroma_adapter_syncs_and_queries_without_external_model(self):
        client = FakeChromaClient()
        store = ChromaResourceStore(
            demo_documents(),
            persist_directory="unused-in-test",
            collection_name="campusmind-test",
            embedder=DeterministicHashEmbedder(128),
            min_score=0.08,
            client=client,
        )
        hits = store.search("校园就业招聘和简历支持", top_k=2)
        self.assertTrue(hits)
        self.assertEqual(hits[0].resource_id, "demo_career_support")
        self.assertEqual(client.metadata["hnsw:space"], "cosine")


if __name__ == "__main__":
    unittest.main(verbosity=2)
