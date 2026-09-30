# CampusMind

CampusMind 是一个面向高校学生学习、求职、情绪支持与校园资源查询场景的风险感知 Multi-Agent 系统。

项目重点不是让 LLM 自由决定所有动作，而是将语义理解、风险策略、Agent 路由、工具执行和结果安全检查拆分为明确的工程边界：LLM 负责理解与生成，确定性的 PolicyEngine 和 Agent Runtime 负责权限与执行约束。

> 项目定位：Agent / LLM Application 工程实践项目。  
> 系统面向非临床校园支持场景，不提供医疗诊断、真实报警或未经核验的学校联系方式。

## Features

### Risk-aware Agent Routing

基于 Intent / Risk 两个独立维度理解请求，由确定性的 `RiskPolicyEngine` 综合生成：

- target Agent
- autonomy level
- mandatory actions
- escalation policy

系统通过 Thin Router 调度：

- `GuidanceAgent`
- `SupportAgent`
- `ResourceAgent`
- `EscalationAgent`
- `GeneralAgent`

高风险请求不会仅依赖 Prompt 约束，而是由程序强制进入 `EscalationAgent` 路径。

针对纯规则 Analyzer 在隐式表达和 hard-negative 场景下的泛化不足，引入：

```text
Rule Fast Path
      ↓
Semantic Fallback
      ↓
Deterministic PolicyEngine
```

Semantic Analyzer 只负责结构化 Intent / Risk 理解，最终 Agent 路由与安全权限仍由确定性代码控制。

在 60 条未参与调参的独立测试样本上：

- Agent Routing Accuracy：`66.7% → 86.7%`
- High-Risk Recall：`71.4% → 100%`

详细评测方法与结果见 [`evaluation/`](./evaluation/)。

### Agent Runtime & Tool Governance

实现统一 `Agent Runtime` 与 `ToolRegistry`，构建受控 Tool Calling Loop。

LLM 可以提出 Tool Call，但不能直接执行工具。Runtime 会在真正执行前检查：

- Tool Scope
- Tool arguments
- MAX_ROUNDS
- duplicate calls
- mandatory tools

从而将：

```text
LLM 动作决策
```

与：

```text
Python 实际执行
```

分离。

### State & Context

使用 Redis 持久化：

- Conversation Memory
- SupportCaseState
- Trace

不同类型状态使用独立 TTL 管理生命周期。

Conversation Memory 保存自然语言聊天历史；`SupportCaseState` 保存当前任务的结构化状态，两者职责分离。

对于：

```text
学校心理咨询中心怎么预约？
→ 我刚才问的事情继续说。
```

这类当前消息语义不完整的模糊续问，系统优先读取当前 CaseState topic，并结合最近用户消息恢复必要上下文，避免错误路由。

显式的新意图始终优先，不会无条件继承历史 Intent。

### Verified Resource RAG

校园资源查询通过 ChromaDB 构建 Vector RAG。

ResourceAgent 使用 `search_verified_resources` 查询经过验证的资源，并将检索结果作为 Evidence 提供给回答链路。

系统遵循：

```text
Verified Resource
      ↓
Vector Index
      ↓
Top-K Retrieval
      ↓
Evidence
      ↓
ResourceAgent
```

当证据不足时，不使用模型参数知识编造具体电话、地址或开放时间。

当前默认使用 `DeterministicHashEmbedder` 保证离线测试可复现，同时通过 `EmbeddingModel` 接口保留后续替换真实 Embedding 模型的能力。

### Safety Guard

安全控制分为执行前、执行中和执行后三层：

```text
Intent / Risk Analysis
        ↓
RiskPolicyEngine
        ↓
Agent Runtime
        ↓
Response Guard
        ↓
Safe Fallback
```

其中：

- PolicyEngine 控制允许的 Agent 和 mandatory actions
- Runtime 控制 Tool 权限与真实执行
- Response Guard 检查最终输出
- Safe Fallback 在高风险输出不满足要求时提供受控回复

### Observability

每次请求生成结构化 Trace，记录：

- request / conversation
- Intent
- Risk
- selected Agent
- Tool executions
- Guard result
- latency
- CaseState

便于定位 Agent 路由、工具执行与安全链路中的问题。

### SSE Demo

提供 FastAPI SSE 接口：

```text
POST /chat/stream
```

事件包括：

```text
accepted
stage
meta
chunk
done
```

Vue 3 Demo 页面可以展示：

- Intent
- Risk
- selected Agent
- Tools
- Guard
- CaseState
- final response

当前 SSE 实现的是 Agent 执行状态事件和最终回答的增量展示，不宣称模型原始 token-level streaming。

---

## Architecture

```text
                     User Request
                          │
                          ▼
                Intent / Risk Analysis
                  │             │
                  │      Semantic Fallback
                  │       when necessary
                  └─────────────┘
                          │
                          ▼
                 RiskPolicyEngine
                          │
              ┌───────────┼───────────┐
              │           │           │
              ▼           ▼           ▼
          target      autonomy    mandatory
           Agent        level       actions
              │
              ▼
                    Thin Router
                          │
                          ▼
                    Agent Runtime
                 ┌────────┴────────┐
                 │                 │
                 ▼                 ▼
               LLM             ToolRegistry
                                   │
                      ┌────────────┴────────────┐
                      ▼                         ▼
                    Redis                    ChromaDB
            Memory / CaseState / Trace        RAG
                      │                         │
                      └────────────┬────────────┘
                                   ▼
                             Agent Result
                                   │
                                   ▼
                            Response Guard
                                   │
                        ┌──────────┴──────────┐
                        ▼                     ▼
                     Passed              Safe Fallback
                        │                     │
                        └──────────┬──────────┘
                                   ▼
                              JSON / SSE
```

## Agent Responsibilities

| Agent | Responsibility | Tool Scope |
| --- | --- | --- |
| GuidanceAgent | 学业、求职与行动建议 | `upsert_support_case` |
| SupportAgent | 非临床情绪支持与关系问题 | `upsert_support_case` |
| ResourceAgent | 校园资源事实查询 | `search_verified_resources` |
| EscalationAgent | 高风险受控处理路径 | Resource + Case tools |
| GeneralAgent | 普通请求兜底 | No tools |

## State & Storage

| Data | Backend | Purpose |
| --- | --- | --- |
| Conversation Memory | Redis List | 最近聊天上下文 |
| SupportCaseState | Redis | 当前任务结构化状态 |
| Trace | Redis | Agent 执行可观测记录 |
| Resource Vector Index | ChromaDB | Verified Resource 检索 |

一句话：

> Redis 管状态，ChromaDB 管知识检索，PolicyEngine / Runtime 管执行权限。

## Tech Stack

- Python
- FastAPI
- Pydantic
- Redis
- ChromaDB
- Docker / Docker Compose
- OpenAI-compatible LLM API
- SSE
- Vue 3
- unittest / FastAPI TestClient

## Project Structure

```text
CampusMind/
├── agents/
│   ├── agent_orchestrator.py
│   ├── base_agent.py
│   ├── factory.py
│   └── tools.py
├── core/
│   ├── intent_analyzer.py
│   ├── risk_analyzer.py
│   ├── semantic_analyzer.py
│   ├── risk_policy.py
│   ├── llm_client.py
│   ├── schemas.py
│   └── config.py
├── state/
├── rag/
├── safety/
├── observability/
├── evaluation/
│   ├── README.md
│   ├── evaluate_routing_risk.py
│   └── datasets/
│       ├── routing_risk_diagnostic.jsonl
│       ├── routing_risk_holdout.jsonl
│       └── holdout_manifest.txt
├── tests/
├── web/
│   └── index.html
├── docs/
├── main.py
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Quick Start

复制环境变量模板：

```powershell
Copy-Item .env.example .env
```

在 `.env` 中配置真实模型参数。

启动 Redis、ChromaDB 与 CampusMind：

```powershell
docker compose build campusmind
docker compose up -d
```

检查服务：

```powershell
Invoke-RestMethod -Uri "http://localhost:8000/health" -Method Get
```

打开 Demo：

```text
http://localhost:8000/demo
```

Swagger：

```text
http://localhost:8000/docs
```

## Run Tests

```powershell
docker compose run --rm --no-deps campusmind python -m unittest discover -s tests -v
```

## Evaluation

Routing / Risk 评测代码与数据位于：

```text
evaluation/
```

支持纯规则与 Hybrid Analyzer 对照评测。

详细说明见：

[`evaluation/README.md`](./evaluation/README.md)

## Design Principles

CampusMind 的核心工程原则：

```text
LLM can propose.
Policy decides.
Runtime executes.
Guard verifies.
Trace records.
```

即：

> 保留 LLM 的语义理解和生成能力，但不把系统最终执行权限完全交给模型。