"""CampusMind IntentAnalyzer：当前先用可解释规则跑通确定性骨架。"""
from __future__ import annotations

from core.schemas import IntentResult, IntentType


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

    def analyze(self, message: str) -> IntentResult:
        text = message.strip().lower()
        scores = {
            intent: sum(1 for keyword in keywords if keyword in text)
            for intent, keywords in self._keywords.items()
        }
        best_intent = max(scores, key=scores.get)
        best_score = scores[best_intent]

        if best_score == 0:
            return IntentResult(intent=IntentType.GENERAL, confidence=0.55)

        confidence = min(0.70 + 0.08 * (best_score - 1), 0.94)
        return IntentResult(intent=best_intent, confidence=confidence)
