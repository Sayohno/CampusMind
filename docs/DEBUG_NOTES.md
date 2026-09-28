# Engineering Debug Notes

## Tool Scope 正确，但 Tool Selection 过度

真实 DeepSeek 测试中，普通“秋招怎么准备？”请求曾调用 `get_support_case`、`search_verified_resources`、`upsert_support_case`。权限没有越界，但开放式指导并不需要查询校园事实，造成额外 Tool round、Token 与延迟。

最终处理：

- GuidanceAgent 只保留 `upsert_support_case`，且 Prompt 明确只有真实状态变化才使用。
- ResourceAgent 独占普通资源检索能力。
- CaseState 已注入 AgentRequest，因此删除普通 Agent 的重复 `get_support_case` 权限。
- GeneralAgent 不暴露任何 Tool。
- 增加 Tool Scope Regression Test 防止后续改动重新放宽权限。

这个案例体现：**权限正确 ≠ 调用行为最优；Tool Scope 是上限，Tool Selection 仍需要 Prompt、Description、Trace 和回归测试共同约束。**

## mandatory tool 重复调用

高风险测试中，Runtime 已执行 `upsert_support_case` 后，LLM 可能再次请求该 Tool。不能简单按工具名禁止第二次调用，因为第二次参数可能用于补充 `next_actions`。

最终处理：使用 `tool_name + normalized arguments` 做精确签名；完全相同的调用跳过，不同参数仍允许执行。
