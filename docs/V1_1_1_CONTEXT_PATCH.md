# V1.1.1 Context-aware Follow-up Patch

## Problem found by real smoke test

Redis persistence was working, but the second turn:

```text
Turn 1: 学校心理咨询中心怎么预约？ -> resource_query / ResourceAgent
Turn 2: 我刚才问的事情继续说。     -> general / GeneralAgent
```

showed an important distinction:

> Saving context is not the same as consuming context.

The old `IntentAnalyzer` classified only the current message. An ambiguous follow-up therefore lost the previous business topic even though Memory and CaseState were still present.

## Fix

`AgentOrchestrator` now loads recent Memory and previous CaseState before intent analysis. `IntentAnalyzer` uses context only when both conditions are true:

1. the current message has no explicit intent keyword;
2. the current message looks like a follow-up (`刚才`, `继续`, `接着`, `然后呢`, etc.).

Fallback order:

```text
explicit current intent
  -> previous CaseState.topic
  -> recent USER memory
  -> General
```

Assistant-generated messages are deliberately ignored as routing evidence, so model output cannot pollute future intent routing.

## Non-goals

This patch is **not** Pattern + Embedding + LLM intent fusion, and it does not add LangGraph. It is a small deterministic context fallback for ambiguous follow-up requests.

## Regression tests

Four tests were added:

- ambiguous resource follow-up stays on `ResourceAgent`;
- an explicit new intent overrides the previous case topic;
- a follow-up with no previous context remains `GeneralAgent`;
- when CaseState is unavailable, recent user memory can provide the previous intent.

Full offline deterministic suite: **56 tests, all passing**.

## Docker smoke test

After replacing the source, rebuild the image:

```powershell
cd D:\CampusMind
docker compose down
docker compose build --no-cache
docker compose up -d
```

Check version:

```powershell
Invoke-RestMethod -Uri "http://localhost:8000/health" -Method Get
```

Expected:

```text
version       : 1.1.1
state_backend : redis
trace_backend : redis
rag_backend   : chroma
```

Then send two turns with the same `user_id` and `conversation_id`:

```text
学校心理咨询中心怎么预约？
我刚才问的事情继续说。
```

The second turn should remain `resource_query` and route to `ResourceAgent`.
