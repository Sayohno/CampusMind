# CampusMind V1 -> V1.1 Upgrade Notes

## Upgrade goal

V1.1 不引入 LangGraph、MCP、Kafka 等额外框架，只解决两个已经明确存在的问题：

1. in-memory state 在进程/容器重启后丢失；
2. keyword RAG 的检索表达能力和持久索引能力有限。

## What changed

### Redis

新增：

- `RedisConversationMemory`
- `RedisCaseStore`
- `RedisTraceStore`
- `state/factory.py`
- `observability/factory.py`

Docker Compose 新增 `redis:7-alpine` 与 named volume `redis_data`。

### Vector RAG / Chroma

新增：

- `EmbeddingModel`
- `DeterministicHashEmbedder`
- `VectorResourceStore`
- `ChromaResourceStore`
- `rag/factory.py`
- named volume `chroma_data`

旧 `VerifiedResourceStore` 仍保留，用作 keyword baseline 与确定性测试。

## Data lifecycle

默认 Demo 配置：

```text
Memory     7 days
CaseState 30 days
Trace      7 days
Chroma     persistent until volume/index is deleted or synchronized
```

TTL 是可配置策略，不是不可修改的业务常量。

## PowerShell upgrade

保留旧 `.env`，替换项目源码后：

```powershell
cd D:\CampusMind
docker compose down
docker compose build --no-cache
docker compose up
```

另开 PowerShell：

```powershell
Invoke-RestMethod -Uri "http://localhost:8000/health" -Method Get
```

期望看到：

```text
version       : 1.1.1
state_backend : redis
trace_backend : redis
rag_backend   : chroma
```

## Persistence smoke test

第一次发请求：

```powershell
$body = @{
    user_id = "persist_user"
    conversation_id = "persist_conv"
    message = "学校心理咨询中心怎么预约？"
} | ConvertTo-Json

$r1 = Invoke-RestMethod `
    -Uri "http://localhost:8000/chat" `
    -Method Post `
    -ContentType "application/json; charset=utf-8" `
    -Body ([System.Text.Encoding]::UTF8.GetBytes($body))

$r1.case_state
```

停止并重新启动：

```powershell
docker compose down
docker compose up
```

然后使用相同 `user_id + conversation_id` 再请求一次。Redis volume 没删除时，Memory / CaseState 不会因为应用容器重建而自动清空。

## Vector RAG smoke test

资源类请求：

```text
学校心理咨询中心怎么预约？
```

应路由至 ResourceAgent，并执行 `search_verified_resources`。Chroma backend 会从 verified corpus 中做向量 Top-K；证据中仍不会出现真实学校电话、地址或时间，因此系统应保持 Grounded Abstention。

## Tests

```powershell
docker compose run --rm campusmind python -m unittest discover -s tests -v
```

V1.1.1 源码当前包含 56 个确定性测试。

## Interview summary

一句话：

> V1 先验证 Agent Runtime 与风险控制链，V1.1 再通过 Redis + TTL 解决跨请求状态持久化，通过 ChromaDB Vector RAG 解决可替换向量检索与持久索引问题，同时保持 Agent / Tool 接口不依赖具体基础设施。


## V1.1.1 final patch: ambiguous follow-up context

Redis 持久化 smoke test 进一步暴露了一个问题：状态已经保存，但 IntentAnalyzer 原先只读取当前一句话。
因此“我刚才问的事情继续说。”会从上一轮 `resource_query` 掉到 `general`。

修复策略不是把全部历史塞进 Prompt，而是只在当前消息没有明确 Intent 且属于 follow-up 时，按下面顺序继承最小必要上下文：

1. previous `CaseState.topic`；
2. recent **user** memory；
3. 仍无依据则保持 `general`。

如果当前消息出现明确的新业务意图，例如“那先不聊这个了，秋招怎么准备？”，当前消息始终覆盖旧 topic。

新增回归覆盖：

- resource query -> ambiguous follow-up 仍路由 ResourceAgent；
- 明确新 Intent 可以覆盖旧 CaseState.topic；
- 新会话无历史时 follow-up 不强猜，保持 GeneralAgent；
- CaseState 不可用时，可从最近 user memory 继承 Intent。
