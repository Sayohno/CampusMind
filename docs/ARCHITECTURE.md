# CampusMind V1.1 Architecture

## 1. Core request path

```text
FastAPI /chat
  -> ChatRequest
  -> Orchestrator
      -> ConversationMemory.get_recent()
      -> CaseStore.get(previous state)
      -> Context-aware IntentAnalyzer
           -> current message explicit intent first
           -> ambiguous follow-up only: previous topic / recent user memory
      -> RiskAnalyzer
      -> RiskPolicyEngine
      -> CaseStore.ensure()
      -> Thin Router
      -> BaseAgent Runtime
           -> mandatory_tools
           -> LLM Tool Calls
           -> Tool Scope / schema / dedup / MAX_ROUNDS
           -> ToolRegistry -> Python Handler
      -> ResponseGuard
      -> ConversationMemory.append_turn()
      -> CaseStore.save()
  -> TraceStore.save()
  -> ChatResponse
```

Agent 层不直接依赖 Redis 或 Chroma。基础设施通过抽象接口注入。

## 2. State backend

### Conversation Memory

```text
ConversationMemory
├─ InMemoryConversationMemory
└─ RedisConversationMemory
```

Redis 结构：

```text
campusmind:memory:{user}:{conversation} -> LIST(JSON ConversationMessage)
```

写入时：`RPUSH -> LTRIM -> EXPIRE`。TTL 在活跃对话每次写入时刷新。

### SupportCaseState

```text
CaseStore
├─ InMemoryCaseStore
└─ RedisCaseStore
```

Redis 结构：

```text
campusmind:case:{user}:{conversation} -> JSON SupportCaseState
```

CaseState 是结构化当前任务状态，不等于聊天记录。V1.1 为了控制范围暂存在 Redis；如果系统需要长期事务、审计和复杂查询，可以保持 `CaseStore` 接口不变替换为关系型数据库。

### Trace

```text
TraceStore
├─ InMemoryTraceStore
└─ RedisTraceStore
```

Redis 结构：

```text
campusmind:trace:item:{request_id} -> JSON TraceRecord + TTL
campusmind:trace:index             -> Sorted Set(timestamp -> request_id)
```

Trace 继续遵守原有隐私边界：不保存原始用户消息，不保存 Tool 参数值。

## 3. Resource RAG backend

```text
ResourceStore
├─ VerifiedResourceStore    # keyword baseline
├─ VectorResourceStore      # pure-Python vector baseline
└─ ChromaResourceStore      # persistent Chroma vector index
```

上层 `search_verified_resources` Tool 只调用：

```python
resource_store.search(query, top_k=...)
```

因此 Agent Runtime 不知道底层到底是 keyword 还是 Chroma。

## 4. Embedding abstraction

```text
EmbeddingModel.embed(texts)
└─ DeterministicHashEmbedder
```

V1.1 的默认 embedder 使用中文 2/3-gram 与英文单词做 signed feature hashing，再做 L2 normalize。它的目标是：

- 不需要联网下载模型；
- 离线单测可复现；
- 真实跑通 vector -> Chroma -> cosine Top-K；
- 后续可以替换成预训练 embedding，而不改 ResourceStore / Tool / Agent。

它不是 BGE、MiniLM 或 OpenAI Embedding，简历与面试必须真实说明。

## 5. Why the architecture does not put Redis/Chroma inside Agent

如果 Agent 直接调用 Redis/Chroma：

- 业务逻辑与基础设施耦合；
- 单元测试需要真实外部服务；
- 将来更换存储要改 Agent；
- Tool / State 权限边界变得模糊。

因此采用：

```text
Agent -> Tool / Store abstraction -> backend implementation
```

这与 CampusMind 原有“LLM 有建议权，Runtime 有执行权”的设计一致。

## 6. Risk path remains unchanged

Redis / Chroma 是基础设施升级，不改变核心安全逻辑：

```text
Intent + Risk
  -> PolicyDecision
  -> allowed_agent / mandatory_tools
  -> Runtime
  -> Guard
  -> Safe Fallback
```

RAG 仍然只回答“事实依据从哪里来”，不能决定“系统允许做什么”。


## 7. Context-aware follow-up routing (V1.1.1)

Redis 解决的是“上下文有没有保存”，但保存不等于消费。真实 smoke test 中：

```text
Turn 1: 学校心理咨询中心怎么预约？ -> resource_query -> ResourceAgent
Turn 2: 我刚才问的事情继续说。     -> 原先会掉到 general
```

V1.1.1 增加最小必要上下文继承：

```text
current message
  -> 有明确 Intent keyword：当前消息优先
  -> 无明确 Intent + 是 follow-up：
       previous CaseState.topic
       -> 若不可用，再看 recent user memory
  -> 仍无依据：General
```

只读取历史 **user** 消息作为路由证据，不使用 assistant 生成文本，避免模型输出反向污染 Intent。
这不是 EchoMind 式 Pattern + Embedding + LLM 三路融合，而是针对模糊续问的确定性 context fallback。
