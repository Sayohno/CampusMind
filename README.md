# CampusMind

CampusMind 是一个面向高校学生学习、求职、情绪支持和校园资源查询场景的 Agent 后端项目。

项目重点不是让 LLM 自由决定所有动作，而是把“理解请求、判断风险、选择 Agent、执行工具、检查输出”拆成几个明确步骤。LLM 主要负责语义理解和生成，最终路由、工具权限和高风险处理由确定性代码约束。

> 这是一个 Agent / LLM Application 工程实践项目，不提供医疗诊断、真实报警或未经核验的学校联系方式。

## 主要功能

### 1. Intent / Risk 路由

请求先经过确定性 `IntentAnalyzer` 和 `RiskAnalyzer`。当规则结果存在明显歧义时，再调用 `SemanticAnalyzer` 做一次语义补充。

```text
Rule Fast Path
      ↓
Semantic Fallback（仅在需要时）
      ↓
RiskPolicyEngine
      ↓
target Agent / mandatory tools / escalation
```

最终 Agent 路由和高风险升级仍由 `RiskPolicyEngine` 决定，Semantic Analyzer 不直接控制工具权限。

当前包含：

- `GuidanceAgent`：学习、求职和行动建议
- `SupportAgent`：非临床情绪支持
- `ResourceAgent`：校园资源查询
- `EscalationAgent`：高风险受控路径
- `GeneralAgent`：普通兜底

### 2. Agent Runtime 与 Tool Calling

所有 Agent 共享同一套 Runtime。模型可以提出 Tool Call，但工具真正执行前会经过：

- Tool Scope 检查
- 参数 Schema 校验
- 最大调用轮次限制
- 重复调用去重
- mandatory tools 强制执行

这样可以把“模型提出动作”和“Python 真正执行动作”分开。

### 3. Redis 状态与上下文

Redis 用于保存三类数据：

- Conversation Memory：近期对话历史
- SupportCaseState：结构化任务状态
- Trace：请求路由、工具执行和耗时等信息

Conversation Memory 和 CaseState 分开管理，并使用不同 TTL。对于“继续刚才的问题”这类当前语义不完整的续问，会结合最近用户消息和 CaseState 恢复必要上下文。

### 4. Verified Resource RAG

校园资源查询通过 ChromaDB 检索经过验证的资源。

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

具体地址、电话、开放时间等事实型信息只基于检索到的 Evidence 回答。证据不足时，系统不会用模型参数知识补齐。

为了保证本地测试可复现，仓库默认使用 `DeterministicHashEmbedder`。Embedding 接口已经抽象，实际部署时可以替换为真实 Embedding 模型。

### 5. 输出安全与 SSE

`ResponseGuard` 在 Agent 生成候选回答后做最终检查。若候选回答不满足当前风险策略或事实约束，则返回 Safe Fallback。

FastAPI 提供：

```text
POST /chat
POST /chat/stream
```

SSE 接口用于推送 Agent 执行阶段和最终回答增量展示。当前实现不是模型原始 token-level streaming。

---

## 请求链路

```text
User Request
    ↓
Conversation Memory / CaseState
    ↓
IntentAnalyzer + RiskAnalyzer
    ↓
Semantic Fallback（按需）
    ↓
RiskPolicyEngine
    ↓
Thin Router
    ↓
Agent Runtime
    ├── LLM
    └── ToolRegistry
          ├── Redis State
          └── ChromaDB RAG
    ↓
ResponseGuard
    ↓
JSON / SSE
```

## 评测

路由与风险评测代码位于 `evaluation/`。

Holdout 数据集在首次正式评测前冻结，共 60 条样本，覆盖：

- clear：24
- implicit：12
- cross_risk：12
- hard_negative：12

在同一批 60 条冻结样本上：

| Metric | Rules | Hybrid |
| --- | ---: | ---: |
| Agent Routing Accuracy | 66.7% (40/60) | 86.7% (52/60) |
| Intent Accuracy | 66.7% | 83.3% |
| Risk Accuracy | 83.3% | 86.7% |

其中 high-risk 样本共 7 条，Rules 识别 5/7，Hybrid 识别 7/7。

这些结果只代表当前自建测试集，不代表公开 Benchmark 或生产环境性能。详细口径、数据分布和运行方式见 [evaluation/README.md](./evaluation/README.md)。

## 技术栈

- Python / FastAPI / Pydantic
- Redis
- ChromaDB
- OpenAI-compatible LLM API
- Docker / Docker Compose
- SSE / Vue 3
- unittest / FastAPI TestClient

## 项目结构

```text
CampusMind/
├── agents/             # Agent Runtime、装配与 Tool Registry
├── core/               # Intent / Risk / Policy / LLM Client
├── state/              # Conversation Memory、CaseState
├── rag/                # Embedding 接口与 ChromaDB 检索
├── safety/             # Response Guard
├── observability/      # Trace
├── evaluation/         # 冻结测试集与评测脚本
├── tests/
├── web/
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

在 `.env` 中配置模型参数后启动：

```powershell
docker compose build campusmind
docker compose up -d
```

健康检查：

```powershell
Invoke-RestMethod -Uri "http://localhost:8000/health" -Method Get
```

Demo：

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

## 当前边界

- 当前 high-risk 流程是受控 Agent 路径，不代表真实人工转介已经发生。
- 默认 Embedding 为可复现的本地实现，不宣称具备生产级语义检索效果。
- Redis、ChromaDB、LLM API 均按单机开发环境设计，未实现多节点高可用。
- 路由与风险指标来自自建冻结测试集，仍需更大规模独立数据继续验证。
