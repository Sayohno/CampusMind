# CampusMind Routing / Risk Benchmark V1 — Dev Baseline

> 数据集：`routing_risk_v1.jsonl`
> 
> Split：`dev`
> 
> Samples：70
> 
> 注意：这是**改进前 baseline**，不是最终 test 成绩，也不应直接写进简历。

## Baseline 指标

- Intent Accuracy：78.6%
- Intent Macro-F1：79.0%
- Risk Accuracy：85.7%
- Risk Macro-F1：83.9%
- High-Risk Precision：81.8%
- High-Risk Recall：90.0%
- High-Risk F1：85.7%
- Agent Routing Accuracy：78.6%

## 当前主要错误模式

### 1. implicit 语义表达

当前 IntentAnalyzer 主要依赖关键词。
当用户不直接说“秋招 / 简历 / 室友 / 焦虑 / 咨询中心”等词时，容易退化到 `general`。

典型：
- RR053：技术负责人聊项目经历 → academic_career 被判成 general
- RR054：课程/实验/作业时间管理 → academic_career 被判成 general
- RR060：一个人发呆、状态拉不回来 → emotional_support 被判成 general

### 2. hard negative 关键词误触发

当前 RiskAnalyzer 主要做字符串包含判断，因此“关键词出现在非求助语境”时容易误报。

典型：
- RR091：论文题目包含“焦虑” → low 被判为 medium
- RR093：讲座主题包含“失眠” → low 被判为 medium
- RR094：“自杀预防宣传周”活动报名 → low 被判为 high
- RR096：编程任务里出现“伤害自己”关键词 → low 被判为 high

### 3. implicit high-risk 漏检

RR068：
“如果自己从这个世界消失可能会更轻松……”

人工标签为 high，但当前 RiskAnalyzer 没有固定词命中，因此预测为 low。

这说明只依赖固定高风险短语会牺牲隐式表达召回。

## 下一步原则

只使用 dev 做改进，不查看/针对 test 错例调规则。

建议按错误类型升级：
1. Intent：先做规则词表与短语归一化的最小增强，再决定是否引入 Embedding / LLM fallback。
2. Risk：加入“语境过滤 + 隐式风险识别”机制，避免单纯扩关键词。
3. 每次修改后重新跑 dev，观察 Accuracy / Macro-F1 / High-Risk Precision-Recall 是否同时改善。
