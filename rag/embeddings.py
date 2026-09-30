"""CampusMind 可替换的向量化接口。

默认提供 DeterministicHashEmbedder：
纯 Python、无需下载外部模型，
适合 Docker Demo、离线测试与可复现回归。

它是轻量 hashing vectorizer，
不冒充深度语义 embedding 模型。

可以保持 embed() 接口不变，
替换为 BGE、sentence-transformers
或外部 Embedding API。
"""

from __future__ import annotations

import hashlib
import math
import re
from abc import ABC, abstractmethod


class EmbeddingModel(ABC):
    @abstractmethod
    def embed(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        raise NotImplementedError


class DeterministicHashEmbedder(
    EmbeddingModel
):
    """字符 n-gram + 单词 hashing 向量。

    输出经过 L2 归一化。

    中文使用 2/3-gram，
    英文与数字使用单词 token。

    该实现没有外部模型下载，
    也不会产生网络调用，
    用于保证 VectorStore / Chroma
    链路能够离线复现。
    """

    def __init__(
        self,
        dimensions: int = 384,
    ) -> None:
        if dimensions < 64:
            raise ValueError(
                "dimensions 必须 >= 64"
            )

        self.dimensions = dimensions

    def embed(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        return [
            self._embed_one(text)
            for text in texts
        ]

    def _embed_one(
        self,
        text: str,
    ) -> list[float]:
        vector = [
            0.0
        ] * self.dimensions

        for token, weight in (
            self._tokens(text)
        ):
            digest = hashlib.blake2b(
                token.encode("utf-8"),
                digest_size=8,
            ).digest()

            raw = int.from_bytes(
                digest,
                "big",
            )

            index = (
                raw % self.dimensions
            )

            sign = (
                1.0
                if (
                    (raw >> 16) & 1
                ) == 0
                else -1.0
            )

            vector[index] += (
                sign * weight
            )

        norm = math.sqrt(
            sum(
                value * value
                for value in vector
            )
        )

        if norm > 0:
            vector = [
                value / norm
                for value in vector
            ]

        return vector

    @staticmethod
    def _tokens(
        text: str,
    ) -> list[tuple[str, float]]:
        normalized = "".join(
            text.lower().split()
        )

        result: list[
            tuple[str, float]
        ] = []

        for word in re.findall(
            r"[a-z0-9_]+",
            normalized,
        ):
            if len(word) >= 2:
                result.append(
                    (
                        f"w:{word}",
                        1.0,
                    )
                )

        chars = [
            char
            for char in normalized
            if (
                "\u4e00"
                <= char
                <= "\u9fff"
            )
        ]

        for n, weight in (
            (2, 1.0),
            (3, 1.25),
        ):
            for index in range(
                len(chars) - n + 1
            ):
                token = "".join(
                    chars[
                        index:
                        index + n
                    ]
                )

                result.append(
                    (
                        f"c{n}:{token}",
                        weight,
                    )
                )

        return result