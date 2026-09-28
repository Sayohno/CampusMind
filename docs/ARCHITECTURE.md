# CampusMind Architecture

## 1. 主调用链

```text
POST /chat
  -> ChatRequest
  -> Conversation Memory + SupportCaseState
  -> IntentAnalyzer
  -> RiskAnalyzer
  -> RiskPolicyEngine
  -> Thin Router
  -> Agent Runtime
  -> mandatory tools / LLM Tool Calling
  -> verified-resource retrieval / Case tools
  -> Response Guard
  -> PASS or Safe Fallback
  -> Memory / CaseState write-back
  -> request Trace
  -> HTTP response
```

## 2. 核心设计

### Intent / Risk 双轴
Intent 回答“用户想解决什么”；Risk 回答“当前风险多高”。相同业务意图可以有不同风险，因此不能只按 Intent 路由。

### Risk-Adaptive Autonomy
低风险保留模型灵活性；高风险由 PolicyEngine 收紧自主权，并由 Runtime 强制执行 mandatory actions。

### Thin Router
PolicyEngine 决定业务/安全边界，Router 只把 `allowed_agent` 映射到具体 Agent，避免路由层变成业务规则中心。

### Tool Scope + mandatory tools
`allowed_tools` 是权限上限；`mandatory_tools` 是当前策略下必须执行的动作。Runtime 是最终执行与权限校验层。

### Memory vs SupportCaseState
Memory 保存“之前聊了什么”；CaseState 保存“当前支持任务处理到哪一步”。CaseState 是结构化业务状态，不是医疗病历。

### RAG / Grounding
ResourceAgent 使用 `search_verified_resources` 获取可核验证据。证据不足时明确无法确认，不用模型参数知识补地址、电话、开放时间或预约方式。

### Response Guard
Policy 管生成前过程；Guard 管生成后出口。高风险或资源事实违反规则时进入 Safe Fallback，不切回更自由的 GeneralAgent。

## 3. 可观测性

每次 `/chat` 生成 `request_id`，Trace 记录 Intent、Risk、Policy Agent、Selected Agent、Tool 执行来源/状态、Guard 与 latency。Trace 不保存原始用户消息和 Tool 参数值。

## 4. 当前边界

- Memory / CaseState / Trace 为 in-memory，容器重启后清空。
- RAG 使用演示知识库，不代表真实高校政策。
- Risk 是内部工程风险分类，不是医学诊断。
- 没有真实人工转介、报警、医疗或热线联动。
- Response Guard 是项目级确定性防线，不应宣传为生产级医疗安全系统。
