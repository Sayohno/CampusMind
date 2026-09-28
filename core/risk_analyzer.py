"""CampusMind RiskAnalyzer。

这里只做内部工程风险信号识别，不做医疗诊断。
"""
from __future__ import annotations

from core.schemas import RiskAssessment, RiskLevel


class RiskAnalyzer:
    _high_signals = [
        "撑不下去", "撑不住", "不想活", "想死", "自杀", "伤害自己", "结束生命", "活不下去"
    ]
    _medium_signals = [
        "焦虑", "睡不着", "失眠", "很难受", "压力很大", "崩溃", "绝望", "连续几天睡不好"
    ]

    def analyze(self, message: str) -> RiskAssessment:
        text = message.strip().lower()

        high_hits = [signal for signal in self._high_signals if signal in text]
        if high_hits:
            return RiskAssessment(
                risk_level=RiskLevel.HIGH,
                signals=high_hits,
                confidence=min(0.90 + 0.02 * (len(high_hits) - 1), 0.98),
            )

        medium_hits = [signal for signal in self._medium_signals if signal in text]
        if medium_hits:
            return RiskAssessment(
                risk_level=RiskLevel.MEDIUM,
                signals=medium_hits,
                confidence=min(0.78 + 0.03 * (len(medium_hits) - 1), 0.92),
            )

        return RiskAssessment(
            risk_level=RiskLevel.LOW,
            signals=[],
            confidence=0.75,
        )
