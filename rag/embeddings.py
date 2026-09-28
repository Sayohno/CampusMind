"""CampusMind 可替换的向量化接口。

V1.1 默认提供 DeterministicHashEmbedder：纯 Python、无需下载外部模型，适合 Docker Demo、
离线测试与可复现回归。它是轻量 hashing vectorizer，不冒充深度语义 embedding 模型。
生产化可保持 embed() 接口不变，替换为 BGE / sentence-transformers / embedding API。
"""
from __future__ import annotations

import hashlib
import math
import re
from abc import ABC, abstractmethod


class EmbeddingModel(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError


class DeterministicHashEmbedder(EmbeddingModel):
    """字符 n-gram + 单词 hashing 向量，L2 归一化。

    目的：让 VectorStore / Chroma 链路完全可离线复现。中文主要使用 1/2/3-gram，
    英文/数字使用单词 token。没有外部模型下载，也不会产生网络调用。
    """

    def __init__(self, dimensions: int = 384) -> None:
        if dimensions < 64:
            raise ValueError("dimensions 必须 >= 64")
        self.dimensions = dimensions

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token, weight in self._tokens(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            raw = int.from_bytes(digest, "big")
            index = raw % self.dimensions
            sign = 1.0 if ((raw >> 16) & 1) == 0 else -1.0
            vector[index] += sign * weight

        norm = math.sqrt(sum(value * value for value in vector))
        if norm > 0:
            vector = [value / norm for value in vector]
        return vector

    @staticmethod
    def _tokens(text: str) -> list[tuple[str, float]]:
        normalized = "".join(text.lower().split())
        result: list[tuple[str, float]] = []

        # 英文/数字词。
        for word in re.findall(r"[a-z0-9_]+", normalized):
            if len(word) >= 2:
                result.append((f"w:{word}", 1.0))

        # 中文字符与 n-gram。过滤常见标点，只保留 CJK 字符。
        chars = [ch for ch in normalized if "\u4e00" <= ch <= "\u9fff"]
        # 中文单字信息量太低，容易让无关文本产生虚假相似度；只使用 2/3-gram。
        for n, weight in ((2, 1.0), (3, 1.25)):
            for idx in range(len(chars) - n + 1):
                result.append((f"c{n}:{''.join(chars[idx: idx + n])}", weight))

        return result
