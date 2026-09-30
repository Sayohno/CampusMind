"""CampusMind 运行配置。

真实 API Key 只从环境变量读取，不写入源码、Dockerfile 或 Git 仓库。
支持本地确定性测试，以及 Redis、ChromaDB 和真实 LLM 后端。
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

from core.llm_client import (
    BaseLLMClient,
    OpenAICompatibleLLMClient,
    StaticLLMClient,
)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, str(default)).strip()
    value = int(raw)
    if value < 0:
        raise ValueError(f"{name} 不能小于 0")
    return value


def _env_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)).strip())


@dataclass(frozen=True)
class Settings:
    # LLM
    llm_mode: str = "static"
    llm_api_key: str = ""
    llm_base_url: str = ""
    llm_model: str = ""
    llm_timeout: float = 60.0
    llm_temperature: float = 0.2
    demo_message: str = "秋招怎么准备？"

    # State / trace backend.
    # Python 单元测试默认使用 in_memory，
    # Docker Compose 可覆盖为 redis。
    state_backend: str = "in_memory"
    trace_backend: str = "in_memory"
    redis_url: str = "redis://localhost:6379/0"
    memory_ttl_seconds: int = 7 * 24 * 60 * 60
    case_ttl_seconds: int = 30 * 24 * 60 * 60
    trace_ttl_seconds: int = 7 * 24 * 60 * 60
    trace_max_items: int = 2000

    # RAG backend.
    # keyword 用于确定性离线测试，
    # Docker Compose 可启用 chroma。
    rag_backend: str = "keyword"
    chroma_persist_dir: str = "./data/chroma"
    chroma_collection: str = "campusmind_resources"
    vector_dimensions: int = 384
    vector_min_score: float = 0.12

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()

        return cls(
            llm_mode=os.getenv(
                "LLM_MODE",
                "static",
            ).strip().lower(),
            llm_api_key=os.getenv(
                "LLM_API_KEY",
                "",
            ).strip(),
            llm_base_url=os.getenv(
                "LLM_BASE_URL",
                "",
            ).strip(),
            llm_model=os.getenv(
                "LLM_MODEL",
                "",
            ).strip(),
            llm_timeout=_env_float(
                "LLM_TIMEOUT",
                60.0,
            ),
            llm_temperature=_env_float(
                "LLM_TEMPERATURE",
                0.2,
            ),
            demo_message=os.getenv(
                "CAMPUSMIND_DEMO_MESSAGE",
                "秋招怎么准备？",
            ).strip(),
            state_backend=os.getenv(
                "STATE_BACKEND",
                "in_memory",
            ).strip().lower(),
            trace_backend=os.getenv(
                "TRACE_BACKEND",
                "in_memory",
            ).strip().lower(),
            redis_url=os.getenv(
                "REDIS_URL",
                "redis://localhost:6379/0",
            ).strip(),
            memory_ttl_seconds=_env_int(
                "MEMORY_TTL_SECONDS",
                7 * 24 * 60 * 60,
            ),
            case_ttl_seconds=_env_int(
                "CASE_TTL_SECONDS",
                30 * 24 * 60 * 60,
            ),
            trace_ttl_seconds=_env_int(
                "TRACE_TTL_SECONDS",
                7 * 24 * 60 * 60,
            ),
            trace_max_items=max(
                1,
                _env_int(
                    "TRACE_MAX_ITEMS",
                    2000,
                ),
            ),
            rag_backend=os.getenv(
                "RAG_BACKEND",
                "keyword",
            ).strip().lower(),
            chroma_persist_dir=os.getenv(
                "CHROMA_PERSIST_DIR",
                "./data/chroma",
            ).strip(),
            chroma_collection=os.getenv(
                "CHROMA_COLLECTION",
                "campusmind_resources",
            ).strip(),
            vector_dimensions=max(
                64,
                _env_int(
                    "VECTOR_DIMENSIONS",
                    384,
                ),
            ),
            vector_min_score=_env_float(
                "VECTOR_MIN_SCORE",
                0.12,
            ),
        )


def build_llm_client(
    settings: Settings,
) -> BaseLLMClient:
    """根据环境配置创建模型适配器。

    static:
        完全离线，不访问真实模型，
        适合确定性测试。

    real:
        调用 OpenAI-compatible
        /chat/completions 协议。
    """

    if settings.llm_mode == "static":
        return StaticLLMClient(
            "[离线 Demo] CampusMind Agent Runtime "
            "已正常完成一次 LLM 调用。"
        )

    if settings.llm_mode != "real":
        raise ValueError(
            "LLM_MODE 只支持 static 或 real"
        )

    missing = [
        name
        for name, value in {
            "LLM_API_KEY": settings.llm_api_key,
            "LLM_BASE_URL": settings.llm_base_url,
            "LLM_MODEL": settings.llm_model,
        }.items()
        if not value
    ]

    if missing:
        raise ValueError(
            "真实模型模式缺少环境变量: "
            + ", ".join(missing)
        )

    return OpenAICompatibleLLMClient(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        model=settings.llm_model,
        timeout=settings.llm_timeout,
        temperature=settings.llm_temperature,
    )