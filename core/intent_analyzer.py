"""CampusMind IntentAnalyzer：确定性规则 + 最小必要上下文继承。

V1.1.1 的目标不是做复杂的三路 Intent Fusion，而是解决一种真实 follow-up：
用户只说“继续刚才那个问题 / 那接下来呢”时，当前句本身没有足够业务语义。
此时只有在“当前句没有明确新意图 + 明显是 follow-up”时，才允许从
CaseState.topic 或最近用户消息继承上一轮 Intent。

显式新意图永远优先，避免旧上下文把用户带偏。
"""
from __future__ import annotations

from core.schemas import ConversationMessage, IntentResult, IntentType


class IntentAnalyzer:
    _keywords: dict[IntentType, list[str]] = {
        IntentType.RESOURCE_QUERY: [
            "咨询中心", "预约", "地址", "开放时间", "几点", "电话", "联系方式", "在哪里"
        ],
        IntentType.HUMAN_SUPPORT: [
            "找个人", "真人", "联系到人", "心理咨询师", "人工帮助"
        ],
        IntentType.ACADEMIC_CAREER: [
            "秋招", "简历", "求职", "面试", "实习", "学业", "考试", "论文", "就业", "考研"
        ],
        IntentType.RELATIONSHIP: [
            "室友", "同学", "朋友", "恋爱", "失恋", "分手", "人际", "关系"
        ],
        IntentType.EMOTIONAL_SUPPORT: [
            "难受", "焦虑", "压力", "睡不好", "失眠", "孤独", "想聊聊", "情绪"
        ],
    }

    _follow_up_markers = (
        "刚才",
        "刚刚",
        "继续",
        "接着",
        "上面",
        "前面",
        "那个问题",
        "这件事",
        "然后呢",
        "接下来呢",
        "下一步呢",
        "继续说",
        "接着说",
    )

    def analyze(
        self,
        message: str,
        *,
        memory_history: list[ConversationMessage] | None = None,
        case_topic: str | None = None,
    ) -> IntentResult:
        text = message.strip().lower()

        # 1) 当前消息有明确业务信号时，永远以当前消息为准。
        current_intent, current_score = self._best_intent(text)
        if current_score > 0:
            return IntentResult(
                intent=current_intent,
                confidence=self._confidence(current_score),
            )

        # 2) 只有“当前句语义不足且明显是 follow-up”时，才继承上下文。
        if self.is_follow_up(message):
            inherited_from_case = self._intent_from_case_topic(case_topic)
            if inherited_from_case is not None:
                return IntentResult(intent=inherited_from_case, confidence=0.80)

            inherited_from_memory = self._intent_from_recent_user_memory(memory_history or [])
            if inherited_from_memory is not None:
                return IntentResult(intent=inherited_from_memory, confidence=0.74)

        # 3) 没有上下文依据时保持 General，绝不强猜。
        return IntentResult(intent=IntentType.GENERAL, confidence=0.55)

    def is_follow_up(self, message: str) -> bool:
        text = message.strip().lower()
        return any(marker in text for marker in self._follow_up_markers)

    def _best_intent(self, text: str) -> tuple[IntentType, int]:
        scores = {
            intent: sum(1 for keyword in keywords if keyword in text)
            for intent, keywords in self._keywords.items()
        }
        best_intent = max(scores, key=scores.get)
        return best_intent, scores[best_intent]

    @staticmethod
    def _confidence(score: int) -> float:
        return min(0.70 + 0.08 * (score - 1), 0.94)

    @staticmethod
    def _intent_from_case_topic(case_topic: str | None) -> IntentType | None:
        if not case_topic:
            return None
        try:
            intent = IntentType(case_topic)
        except ValueError:
            return None
        return None if intent == IntentType.GENERAL else intent

    def _intent_from_recent_user_memory(
        self,
        memory_history: list[ConversationMessage],
    ) -> IntentType | None:
        # 只看用户历史，不使用 assistant 回复做路由证据，避免模型生成文本反向污染 Intent。
        for item in reversed(memory_history):
            if item.role != "user":
                continue
            intent, score = self._best_intent(item.content.strip().lower())
            if score > 0:
                return intent
        return None
