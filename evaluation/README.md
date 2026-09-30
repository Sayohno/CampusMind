# Routing & Risk Evaluation

CampusMind 对以下三个部分进行独立评测：

- Intent Classification
- Risk Classification
- Agent Routing

评测不会真正运行 Agent、Tool Calling、RAG 或最终回答生成。

---

## Evaluation Modes

### Rules

只使用确定性的：

```text
IntentAnalyzer
+
RiskAnalyzer
+
RiskPolicyEngine
```

### Hybrid

使用：

```text
Rule Fast Path
      ↓
Semantic Fallback
      ↓
RiskPolicyEngine
```

SemanticAnalyzer 仅用于处理规则结果中的语义不确定性。

最终 Agent 路由、高风险升级和权限控制仍由确定性 PolicyEngine 完成。

---

## Datasets

### Diagnostic

```text
datasets/routing_risk_diagnostic.jsonl
```

用于开发阶段分析：

- clear
- implicit
- cross_risk
- hard_negative

该数据集已经参与开发过程，因此不作为最终泛化能力的主要依据。

### Holdout

```text
datasets/routing_risk_holdout.jsonl
```

包含 60 条未参与 Analyzer 和 Prompt 调参的独立测试样本。

类别覆盖：

- clear
- implicit
- cross_risk
- hard_negative

最终评测完成后，不再根据 Holdout 错误继续调整规则或 Prompt。

数据集冻结信息位于：

```text
datasets/holdout_manifest.txt
```

---

## Metrics

主要指标：

- Intent Accuracy
- Intent Macro-F1
- Risk Accuracy
- Risk Macro-F1
- High-Risk Precision
- High-Risk Recall
- High-Risk F1
- Agent Routing Accuracy

对于风险感知系统，High-Risk Recall 用于衡量真实高风险请求是否发生漏检。

---

## Holdout Result

60 条独立测试样本：

| Metric | Rules | Hybrid |
| --- | ---: | ---: |
| Intent Accuracy | 66.7% | 83.3% |
| Intent Macro-F1 | 66.5% | 82.8% |
| Risk Accuracy | 83.3% | 86.7% |
| Risk Macro-F1 | 77.3% | 83.6% |
| High-Risk Precision | 100.0% | 100.0% |
| High-Risk Recall | 71.4% | 100.0% |
| High-Risk F1 | 83.3% | 100.0% |
| Agent Routing Accuracy | 66.7% | 86.7% |

Hybrid Analyzer 在该测试集上：

- Agent Routing Accuracy：`66.7% → 86.7%`
- High-Risk Recall：`71.4% → 100.0%`

这些结果仅代表当前自建测试集，不代表公开标准 Benchmark 或生产环境性能。

---

## Run

规则模式：

```powershell
python evaluation/evaluate_routing_risk.py `
    --dataset diagnostic `
    --split all `
    --mode rules
```

Hybrid 模式：

```powershell
python evaluation/evaluate_routing_risk.py `
    --dataset diagnostic `
    --split all `
    --mode hybrid
```

Hybrid 需要：

```text
LLM_MODE=real
LLM_API_KEY
LLM_BASE_URL
LLM_MODEL
```

正式测试集同样可以通过：

```text
--dataset holdout
```

选择。

生成的 `report_*.json` 为本地评测产物，不提交 Git。