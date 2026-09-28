"""CampusMind 运行配置。

真实 API Key 只从环境变量读取，不写进源码、Dockerfile 或 Git 仓库。
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

from core.llm_client import BaseLLMClient, OpenAICompatibleLLMClient, StaticLLMClient


@dataclass(frozen=True)
class Settings:
    llm_mode: str = "static"
    llm_api_key: str = ""
    llm_base_url: str = ""
    llm_model: str = ""
    llm_timeout: float = 60.0
    llm_temperature: float = 0.2
    demo_message: str = "秋招怎么准备？"

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        return cls(
            llm_mode=os.getenv("LLM_MODE", "static").strip().lower(),
            llm_api_key=os.getenv("LLM_API_KEY", "").strip(),
            llm_base_url=os.getenv("LLM_BASE_URL", "").strip(),
            llm_model=os.getenv("LLM_MODEL", "").strip(),
            llm_timeout=float(os.getenv("LLM_TIMEOUT", "60")),
            llm_temperature=float(os.getenv("LLM_TEMPERATURE", "0.2")),
            demo_message=os.getenv("CAMPUSMIND_DEMO_MESSAGE", "秋招怎么准备？").strip(),
        )


def build_llm_client(settings: Settings) -> BaseLLMClient:
    """根据环境配置创建模型适配器。

    static: 完全离线，不访问网络，适合确定性测试。
    real: 调用 OpenAI-compatible /chat/completions 协议。
    """
    if settings.llm_mode == "static":
        return StaticLLMClient("[离线 Demo] CampusMind Agent Runtime 已正常完成一次 LLM 调用。")

    if settings.llm_mode != "real":
        raise ValueError("LLM_MODE 只支持 static 或 real")

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
        raise ValueError(f"真实模型模式缺少环境变量: {', '.join(missing)}")

    return OpenAICompatibleLLMClient(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        model=settings.llm_model,
        timeout=settings.llm_timeout,
        temperature=settings.llm_temperature,
    )
