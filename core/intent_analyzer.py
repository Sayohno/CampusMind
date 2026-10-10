"""Deterministic intent analysis with optional semantic fallback signals."""
from __future__ import annotations

from core.intent_rules import (
    CAREER_GOAL_MARKERS,
    FOLLOW_UP_MARKERS,
    HUMAN_GOAL_MARKERS,
    RELATIONSHIP_GOAL_MARKERS,
    RESOURCE_GOAL_MARKERS,
    SIGNALS,
    TIE_PRIORITY,
)
from core.schemas import ConversationMessage, IntentResult, IntentType


class IntentAnalyzer:
    _signals = SIGNALS
    _resource_goal_markers = RESOURCE_GOAL_MARKERS
    _career_goal_markers = CAREER_GOAL_MARKERS
    _human_goal_markers = HUMAN_GOAL_MARKERS
    _relationship_goal_markers = RELATIONSHIP_GOAL_MARKERS
    _tie_priority = TIE_PRIORITY
    _follow_up_markers = FOLLOW_UP_MARKERS

    def analyze(
        self,
        message: str,
        *,
        memory_history: list[ConversationMessage] | None = None,
        case_topic: str | None = None,
    ) -> IntentResult:
        text = message.strip().lower()
        current_intent, current_score = self._best_intent(text)

        if current_score > 0:
            return IntentResult(
                intent=current_intent,
                confidence=self._confidence(current_score),
            )

        # Only inherit context for explicit follow-ups. A new explicit intent wins.
        if self.is_follow_up(message):
            inherited_from_case = self._intent_from_case_topic(case_topic)
            if inherited_from_case is not None:
                return IntentResult(intent=inherited_from_case, confidence=0.80)

            inherited_from_memory = self._intent_from_recent_user_memory(
                memory_history or []
            )
            if inherited_from_memory is not None:
                return IntentResult(intent=inherited_from_memory, confidence=0.74)

        return IntentResult(intent=IntentType.GENERAL, confidence=0.55)

    def is_follow_up(self, message: str) -> bool:
        text = message.strip().lower()
        return any(marker in text for marker in self._follow_up_markers)

    def needs_semantic_fallback(
        self,
        message: str,
        result: IntentResult,
    ) -> bool:
        """Return True when deterministic intent evidence is weak or conflicting."""
        if self.is_follow_up(message):
            return False

        text = message.strip().lower()
        scores = self._score_intents(text)
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)

        if result.intent == IntentType.GENERAL and result.confidence <= 0.60:
            return True

        positive = [(intent, score) for intent, score in ranked if score > 0]
        general_score = scores.get(IntentType.GENERAL, 0.0)

        if general_score > 0 and any(
            score > 0
            for intent, score in scores.items()
            if intent != IntentType.GENERAL
        ):
            return True

        if len(positive) >= 2:
            margin = positive[0][1] - positive[1][1]
            if margin <= 1.5:
                return True

        return False

    def _score_intents(self, text: str) -> dict[IntentType, float]:
        scores = {intent: 0.0 for intent in self._signals}

        for intent, signals in self._signals.items():
            for phrase, weight in signals.items():
                if phrase in text:
                    scores[intent] += weight

        if any(marker in text for marker in self._resource_goal_markers):
            scores[IntentType.RESOURCE_QUERY] += 2.0

        if "有没有" in text and any(
            noun in text
            for noun in ("讲座", "活动", "资源", "中心", "热线", "通知")
        ):
            scores[IntentType.RESOURCE_QUERY] += 2.0

        if any(marker in text for marker in self._career_goal_markers):
            scores[IntentType.ACADEMIC_CAREER] += 1.8

        if any(marker in text for marker in self._human_goal_markers):
            scores[IntentType.HUMAN_SUPPORT] += 0.2

        if any(marker in text for marker in self._relationship_goal_markers):
            scores[IntentType.RELATIONSHIP] += 1.5

        return scores

    def _best_intent(self, text: str) -> tuple[IntentType, float]:
        scores = self._score_intents(text)
        best_intent = max(
            self._tie_priority,
            key=lambda intent: scores[intent],
        )
        return best_intent, scores[best_intent]

    @staticmethod
    def _confidence(score: float) -> float:
        return min(0.66 + 0.06 * score, 0.94)

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
        # Use user messages only so generated assistant text cannot affect routing.
        for item in reversed(memory_history):
            if item.role != "user":
                continue

            intent, score = self._best_intent(item.content.strip().lower())
            if score > 0:
                return intent

        return None
