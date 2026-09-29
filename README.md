# CampusMind

CampusMind 是一个面向高校学生**非临床支持场景**的 Risk-aware Multi-Agent 系统。它不是医疗诊断系统，而是通过 Intent / Risk 双轴理解、确定性 Policy、多 Agent 路由、受控 Tool Calling、Memory / SupportCaseState、Verified-resource RAG、Response Guard、Trace 与回归测试，将 LLM 的灵活性限制在可解释、可测试的工程边界内。

## 当前实现

1. **Redis State Backend**：Conversation Memory、SupportCaseState 与 Trace 使用 Redis 存储，并通过不同 TTL 管理生命周期；结合 Docker Volume，使容器重建后状态仍可恢复。

2. **ChromaDB Vector RAG**：保留统一 ResourceStore / Embedding 接口，将经过验证的校园资源写入 ChromaDB 向量检索后端，由 `search_verified_resources` Tool 返回 Top-K evidence；生成阶段要求基于检索证据回答，证据不足时不补全未经验证的具体事实。

3. **Context-aware Follow-up Routing**：对于“继续刚才的问题”“那接下来呢”等缺少独立业务语义的 follow-up 请求，IntentAnalyzer 会结合 previous CaseState.topic 与 recent user memory 恢复上一轮业务语义；若当前消息包含明确新意图，则始终优先采用当前意图。

4. **Risk-aware Agent Runtime**：基于 Intent / Risk 双轴分析和确定性 RiskPolicyEngine 控制 Agent、Tool Scope 与 mandatory actions；低风险场景保留模型在授权范围内的自主性，高风险场景由 Runtime 收紧权限并进入更确定性的处理路径。

5. **Safety & Observability**：通过 Response Guard、Safe Fallback、Trace 和 Golden Cases 对最终输出、安全策略与执行链路进行校验，并记录 Intent、Risk、Agent、Tool、Guard、latency 等运行信息。

> **项目定位：** 学习 / 作品集 / 实习面试 Demo。资源库仍为演示数据；不提供真实人工转介、报警、医疗诊断或真实学校联系方式。
## Highlights

- **Risk-Adaptive Autonomy**：低风险保留模型灵活性，高风险由 PolicyEngine 收紧自主权并强制安全动作。
- **Intent / Risk 双轴**：业务方向与风险程度分开建模，同一求职意图可以走普通 Guidance，也可以因高风险切到 Escalation。
- **Context-aware Follow-up**：模糊续问只继承最小必要上下文（previous case topic + recent user memory），显式新意图不会被旧状态覆盖。
- **Deterministic Policy + Thin Router**：安全规则由 Python 决定，Router 只负责组件映射。
- **Runtime-controlled Tool Calling**：LLM 产生 ToolCall，Runtime 做 Tool Scope / 参数 / MAX_ROUNDS / dedup 控制，Python Handler 真正执行。
- **Redis-backed State**：Memory、CaseState、Trace 通过 Store abstraction 与业务逻辑解耦，Docker 下默认使用 Redis。
- **TTL Lifecycle**：Memory 默认 7 天、CaseState 默认 30 天、Trace 默认 7 天；均可通过环境变量修改，`0` 表示不设置过期。
- **ChromaDB Vector RAG**：`ResourceStore` 可在 keyword / in-memory vector / Chroma 三种后端间切换。
- **Verified-resource Grounding**：只索引 `verified=true` 的资源；检索不到可靠证据时必须 abstain，不用模型参数知识补电话、地址、开放时间。
- **Response Guard + Safe Fallback**：Policy 管过程，Guard 管出口；高风险失败后继续收紧，不回退到自由 Agent。
- **Observability**：每个 `/chat` 请求生成 `request_id`，Trace 可定位 Intent、Risk、Agent、Tool、Guard 与 latency。
- **Golden Cases / Regression Tests**：当前 **56 个离线确定性测试全部通过**；测试默认不消耗真实 LLM API。

## Architecture

```text
POST /chat
  -> ChatRequest
  -> Conversation Memory --------> Redis (TTL)
  -> SupportCaseState ------------> Redis (longer TTL)
  -> Context-aware IntentAnalyzer
       -> current message first
       -> ambiguous follow-up: previous case topic / recent user memory
  -> RiskAnalyzer
  -> RiskPolicyEngine
  -> Thin Router
  -> Agent Runtime
       -> mandatory tools
       -> LLM Tool Calling
       -> Tool Scope / dedup / MAX_ROUNDS
       -> Python Handler
            -> Case Tool ----------> Redis
            -> RAG Tool -----------> ResourceStore
                                      ├─ keyword baseline
                                      ├─ in-memory vector
                                      └─ ChromaDB persistent vector index
  -> Response Guard
       -> PASS / Safe Fallback
  -> Memory / CaseState write-back
  -> Trace ------------------------> Redis (TTL)
  -> HTTP response
```

更详细设计见 [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)，V1.1 升级说明见 [`docs/V1_1_UPGRADE.md`](docs/V1_1_UPGRADE.md)。

## Why Redis and Chroma are different

Redis 和 Chroma 解决的不是同一类问题：

- **Redis**：保存“系统当前状态”——对话历史、CaseState、Trace、TTL；
- **ChromaDB**：保存“可检索知识的向量索引”——根据 query vector 找 Top-K evidence；
- **Policy / Guard**：仍负责安全边界，不能被 Redis 或 RAG 替代。

一句话：**Redis 管状态，Chroma 管知识检索，Policy 管权限。**

## State lifecycle

Docker 默认配置：

| 数据 | 后端 | 默认 TTL | 说明 |
|---|---|---:|---|
| Conversation Memory | Redis List | 7 天 | 每次写入刷新 TTL，只保留最近 N 条 |
| SupportCaseState | Redis JSON String | 30 天 | 当前 Demo 的业务状态；生产化可迁移关系型数据库 |
| Trace | Redis String + Sorted Set | 7 天 | Trace payload 到期清除，索引限制最大条数 |
| Vector Index | ChromaDB persistent volume | 无 TTL | 由 corpus 同步更新 |

TTL 都是 Demo 默认值，不是业务硬编码。通过 `.env` 可修改。

## Vector RAG design

V1.1 的 Vector RAG 流程：

```text
verified resource corpus
  -> embedding_text(title + boosted verified keywords + content)
  -> DeterministicHashEmbedder
  -> Chroma cosine index

query
  -> same embedder
  -> Chroma Top-K
  -> score threshold
  -> evidence
  -> LLM grounded generation
  -> Response Guard
```

### Important boundary

`DeterministicHashEmbedder` 是为了**离线可复现、无需下载外部 embedding 模型**而实现的轻量 hashing vectorizer，不应包装成 BGE / sentence-transformers 等预训练语义模型。它的接口是可替换的：生产化可换成 BGE、sentence-transformers 或外部 embedding API，而不修改 Agent Runtime / Tool 接口。

因此 V1.1 可以真实描述为：

> 使用 ChromaDB 构建持久化 Vector RAG，并通过可替换 EmbeddingModel 接口隔离向量化实现。

不要描述为：

> 已使用 BGE / OpenAI Embedding（当前代码没有）。

## Agent responsibilities

| Agent | 主要职责 | Tool Scope |
|---|---|---|
| GuidanceAgent | 学业 / 求职 / 行动建议 | `upsert_support_case`（仅明确状态变化时） |
| SupportAgent | 非临床情绪支持与澄清 | `upsert_support_case` |
| ResourceAgent | 可核验校园资源事实 | `search_verified_resources` |
| EscalationAgent | 高风险受控处理路径 | `search_verified_resources`, `upsert_support_case` |
| GeneralAgent | 普通兜底 | 无 Tool |

Tool Scope 是**最小权限设计**。普通“秋招怎么准备？”不会向 GuidanceAgent 暴露 RAG Tool；涉及校园事实时由 ResourceAgent 处理。

## Tech stack

- Python 3.11
- FastAPI + Uvicorn
- Pydantic v2
- HTTPX
- Redis / redis-py
- ChromaDB
- Docker / Docker Compose
- OpenAI-compatible Chat Completions API（可接 DeepSeek 等）
- `unittest` + FastAPI TestClient

## Project structure

```text
CampusMind/
├─ main.py
├─ Dockerfile
├─ docker-compose.yml
├─ requirements.txt
├─ .env.example
├─ agents/
├─ core/
├─ state/
│  ├─ memory_store.py       # in-memory / Redis Memory
│  ├─ case_store.py         # in-memory / Redis CaseState
│  └─ factory.py
├─ observability/
│  ├─ trace_store.py        # in-memory / Redis Trace
│  └─ factory.py
├─ rag/
│  ├─ embeddings.py         # 可替换 EmbeddingModel + 离线 hash embedder
│  ├─ resource_store.py     # keyword / in-memory vector
│  ├─ chroma_store.py       # Chroma adapter
│  ├─ factory.py
│  └─ corpus/demo_resources.json
├─ safety/
├─ docs/
└─ tests/
```

## Quick start with Docker

### 1. 保留你的真实 `.env`

如果从 V1 升级，不要把真实 API Key 放进 `.env.example`。你的本地 `.env` 可以继续使用；即使没有新增 Redis / RAG 配置，Docker Compose 也会默认启用 Redis + Chroma。

推荐 `.env`：

```env
LLM_MODE=real
LLM_API_KEY=你的真实Key
LLM_BASE_URL=https://api.deepseek.com
LLM_MODEL=deepseek-flash
LLM_TIMEOUT=60
LLM_TEMPERATURE=0.2

STATE_BACKEND=redis
TRACE_BACKEND=redis
REDIS_URL=redis://redis:6379/0
MEMORY_TTL_SECONDS=604800
CASE_TTL_SECONDS=2592000
TRACE_TTL_SECONDS=604800

RAG_BACKEND=chroma
CHROMA_PERSIST_DIR=/app/data/chroma
```

### 2. 构建并启动

PowerShell：

```powershell
cd D:\CampusMind
docker compose down
docker compose build --no-cache
docker compose up
```

首次升级会额外拉取 Redis 镜像并安装 `redis` / `chromadb` Python 依赖。

启动后：

- API: `http://localhost:8000`
- Swagger: `http://localhost:8000/docs`
- Health: `http://localhost:8000/health`

Health 正常情况下会显示：

```text
version       : 1.1.1
state_backend : redis
trace_backend : redis
rag_backend   : chroma
```

### 3. 停止但保留状态

```powershell
docker compose down
```

Redis / Chroma 使用 named volumes，普通 `down` 后数据仍保留。

**不要**随便执行：

```powershell
docker compose down -v
```

`-v` 会删除 Redis / Chroma volumes，相当于主动清除持久化数据。

## API examples

### Health

```powershell
Invoke-RestMethod -Uri "http://localhost:8000/health" -Method Get
```

### Chat

```powershell
$body = @{
    user_id = "demo_user"
    conversation_id = "demo_conv_1"
    message = "学校心理咨询中心怎么预约？"
} | ConvertTo-Json

$result = Invoke-RestMethod `
    -Uri "http://localhost:8000/chat" `
    -Method Post `
    -ContentType "application/json; charset=utf-8" `
    -Body ([System.Text.Encoding]::UTF8.GetBytes($body))

$result
```

### Trace

```powershell
$requestId = $result.request_id
Invoke-RestMethod -Uri "http://localhost:8000/traces/$requestId" -Method Get
```
