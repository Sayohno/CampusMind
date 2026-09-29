# SSE + Thin Vue Demo

## 为什么加这一层

CampusMind 的核心仍然是 Agent Runtime，而不是前端。增加 SSE 与薄 Vue 页面主要解决两个问题：

1. 用户不必只通过 PowerShell / Swagger 调接口，可以直接看到聊天结果；
2. Agent 的 Intent、Risk、Selected Agent、Tools、Guard、CaseState 可以作为结构化 metadata 展示，而不是只显示最终文本。

## 为什么没有上完整 Vue CLI / Vite 工程

当前目标是 Agent / Python 后端实习展示，不是前端工程岗位。因此只用一个 `web/index.html` + Vue CDN：

- 不引入 Node/npm 构建链；
- FastAPI 同源提供 `/demo`；
- 不需要额外 CORS；
- 前端只负责展示，不侵入 Orchestrator / Runtime。

## 为什么 POST SSE 不使用 EventSource

浏览器原生 `EventSource` 主要用于 GET 请求，难以直接发送 JSON body。CampusMind 的 ChatRequest 包含 `user_id / conversation_id / message`，因此 `/chat/stream` 使用 POST，并由前端通过 `fetch()` 读取 `ReadableStream`，按 SSE 的 `event:` / `data:` 格式解析。

这仍然是标准 `text/event-stream` 传输，只是客户端不是 `EventSource` API。

## 当前 SSE 事件

```text
accepted -> 请求已建立，返回 request_id
stage    -> 当前进入 Agent orchestrating 阶段
meta     -> intent / risk / agent / tools / guard / latency
chunk    -> 最终回复文本分块
done     -> case_state / trace_url
error    -> 流已经建立后发生错误时的结构化错误事件
```

## 重要边界：当前不是 token-level LLM streaming

当前 `OpenAICompatibleLLMClient.chat()` 仍使用普通非流式 `/chat/completions`。因此 Agent Runtime 与 Tool Calling 全部完成后，后端才拿到最终回答，再通过 `chunk` 事件增量传给前端。

这能验证：

- FastAPI `StreamingResponse`；
- `text/event-stream`；
- SSE framing；
- 浏览器 ReadableStream；
- Vue 增量渲染；
- Agent metadata streaming；
- Trace 与 SSE 链路集成。

但它**不会降低模型 first-token latency**。

如果未来要做真正 token-level streaming，需要进一步处理：

- LLM provider `stream=true`；
- SSE / chunk delta 解析；
- streaming tool_calls 参数拼接；
- Tool Calling 轮次结束检测；
- Guard 与最终输出流的安全关系。

当前项目选择在这里封版，避免为了一个简历关键词破坏已经稳定的 Runtime。

## 面试话术

> 我给 CampusMind 增加了一层很薄的 Vue + SSE 展示。前端不参与 Agent 决策，只把 Runtime 的 metadata 和最终回复以事件流展示出来。因为 Chat API 是 POST JSON，我没有直接用 EventSource，而是用 fetch 读取 `text/event-stream` 并解析 SSE framing。当前模型调用本身仍是非流式，所以我会明确称它为 Agent event streaming，而不是 token-level LLM streaming。
