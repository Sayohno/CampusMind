# CampusMind Validation

本文档记录 CampusMind 的主要验证方式。

## Automated Tests

在 Docker 环境运行完整回归测试：

```powershell
docker compose build campusmind
docker compose run --rm --no-deps campusmind python -m unittest discover -s tests -v
```

测试覆盖：

- Intent / Risk / Policy
- Agent Routing
- Tool Scope
- mandatory tools
- Agent Runtime
- Tool Call deduplication
- Response Guard
- Safe Fallback
- Conversation Memory
- SupportCaseState
- Context-aware follow-up
- Redis backends
- Vector RAG
- ChromaDB adapter
- Semantic Analyzer
- Hybrid Analyzer
- HTTP API
- SSE
- Trace

## Health Check

启动服务：

```powershell
docker compose up -d
```

检查：

```powershell
Invoke-RestMethod `
    -Uri "http://localhost:8000/health" `
    -Method Get
```

Docker 部署下应确认：

- API 可访问
- Redis backend 正常
- ChromaDB backend 正常

## Representative Requests

### Academic / Career

```text
下周有技术面试，我该怎么准备？
```

预期：

```text
GuidanceAgent
```

### Resource Query

```text
学校心理咨询中心怎么预约？
```

预期：

```text
ResourceAgent
```

并使用：

```text
search_verified_resources
```

### High Risk

高风险测试请求应验证：

```text
Risk = high
→ EscalationAgent
→ mandatory tools
```

高风险升级不能只依赖 Prompt。

### Context Follow-up

第一轮：

```text
学校心理咨询中心怎么预约？
```

第二轮：

```text
我刚才问的事情继续说。
```

第二轮应通过 CaseState / recent user context 恢复必要的资源查询语义。

如果第二轮已经表达新的明确 Intent，则新 Intent 应覆盖之前上下文。

## Routing / Risk Evaluation

评测脚本：

```text
evaluation/evaluate_routing_risk.py
```

开发诊断集：

```powershell
python evaluation/evaluate_routing_risk.py `
    --dataset diagnostic `
    --split all `
    --mode rules
```

Hybrid：

```powershell
python evaluation/evaluate_routing_risk.py `
    --dataset diagnostic `
    --split all `
    --mode hybrid
```

最终独立测试数据位于：

```text
evaluation/datasets/routing_risk_holdout.jsonl
```

详细评测方法和最终结果见：

```text
evaluation/README.md
```

最终测试集用于评估泛化能力，不用于继续针对具体错误调规则或 Prompt。