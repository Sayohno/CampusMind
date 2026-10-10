"""Deterministic risk analysis with semantic-fallback hints."""
from __future__ import annotations

import re

from core.risk_rules import (
    EXPLICIT_IMPAIRMENT_MARKERS,
    FUNCTIONAL_SIGNALS,
    HIGH_SIGNALS,
    INDIRECT_HIGH_SIGNALS,
    MEDIUM_SIGNALS,
    META_CONTEXT_MARKERS,
    PERSONAL_DISTRESS_MARKERS,
    RELIEF_MARKERS,
    TEMPORAL_OR_PERSONAL_MARKERS,
)
from core.schemas import RiskAssessment, RiskLevel


class RiskAnalyzer:
    _high_signals = HIGH_SIGNALS
    _indirect_high_signals = INDIRECT_HIGH_SIGNALS
    _medium_signals = MEDIUM_SIGNALS
    _functional_signals = FUNCTIONAL_SIGNALS
    _meta_context_markers = META_CONTEXT_MARKERS
    _personal_distress_markers = PERSONAL_DISTRESS_MARKERS
    _relief_markers = RELIEF_MARKERS

    def analyze(self, message: str) -> RiskAssessment:
        text = message.strip().lower()
        risk_text = self._remove_quoted_text(text)

        meta_context = self._is_meta_context(text)
        personal_distress = self._has_personal_distress(risk_text)

        # Quoted/research/code contexts are not treated as personal risk by default.
        if meta_context and not personal_distress:
            return RiskAssessment(
                risk_level=RiskLevel.LOW,
                signals=["contextual_non_personal_reference"],
                confidence=0.88,
            )

        high_hits = [
            signal for signal in self._high_signals if signal in risk_text
        ]
        if high_hits:
            return RiskAssessment(
                risk_level=RiskLevel.HIGH,
                signals=high_hits,
                confidence=min(0.90 + 0.02 * (len(high_hits) - 1), 0.98),
            )

        indirect_high_hits = [
            signal
            for signal in self._indirect_high_signals
            if signal in risk_text
        ]
        if indirect_high_hits and personal_distress:
            relief_hits = [
                marker for marker in self._relief_markers if marker in risk_text
            ]
            return RiskAssessment(
                risk_level=RiskLevel.HIGH,
                signals=indirect_high_hits + relief_hits,
                confidence=0.92 if relief_hits else 0.88,
            )

        medium_hits = [
            signal for signal in self._medium_signals if signal in risk_text
        ]
        if medium_hits:
            return RiskAssessment(
                risk_level=RiskLevel.MEDIUM,
                signals=medium_hits,
                confidence=min(0.78 + 0.03 * (len(medium_hits) - 1), 0.92),
            )

        functional_hits = [
            signal for signal in self._functional_signals if signal in risk_text
        ]
        has_explicit_impairment = any(
            marker in risk_text for marker in EXPLICIT_IMPAIRMENT_MARKERS
        )

        if len(functional_hits) >= 2 or has_explicit_impairment:
            return RiskAssessment(
                risk_level=RiskLevel.MEDIUM,
                signals=functional_hits,
                confidence=min(0.72 + 0.04 * len(functional_hits), 0.88),
            )

        return RiskAssessment(
            risk_level=RiskLevel.LOW,
            signals=[],
            confidence=0.75,
        )

    def needs_semantic_fallback(
        self,
        message: str,
        result: RiskAssessment,
    ) -> bool:
        """Return True when rule-based risk evidence needs semantic disambiguation."""
        text = message.strip().lower()

        high_hits = [signal for signal in self._high_signals if signal in text]
        indirect_high_hits = [
            signal for signal in self._indirect_high_signals if signal in text
        ]
        medium_hits = [
            signal for signal in self._medium_signals if signal in text
        ]

        if self._is_meta_context(text) and (
            high_hits or indirect_high_hits or medium_hits
        ):
            return True

        if result.risk_level == RiskLevel.LOW and (
            high_hits or indirect_high_hits
        ):
            return True

        functional_hits = [
            signal for signal in self._functional_signals if signal in text
        ]
        temporal_or_personal = any(
            marker in text for marker in TEMPORAL_OR_PERSONAL_MARKERS
        )

        return (
            result.risk_level == RiskLevel.LOW
            and bool(functional_hits)
            and temporal_or_personal
        )

    def has_potential_high_signal(self, message: str) -> bool:
        """Check raw text for a possible high-risk signal after fallback failure."""
        text = message.strip().lower()
        return any(
            signal in text
            for signal in (*self._high_signals, *self._indirect_high_signals)
        )

    def _is_meta_context(self, text: str) -> bool:
        return any(marker in text for marker in self._meta_context_markers)

    def _has_personal_distress(self, text: str) -> bool:
        return any(marker in text for marker in self._personal_distress_markers)

    @staticmethod
    def _remove_quoted_text(text: str) -> str:
        patterns = (
            r"“[^”]*”",
            r"‘[^’]*’",
            r'"[^"]*"',
            r"'[^']*'",
        )

        result = text
        for pattern in patterns:
            result = re.sub(pattern, " ", result)
        return result
