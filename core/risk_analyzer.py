"""CampusMind RiskAnalyzer：基于风险信号、语境与功能影响的确定性风险识别。

这里只进行系统内部工程风险分级，不进行医疗诊断。

设计目标：
1. 保留明确高风险表达的高召回；
2. 识别部分不包含固定关键词的隐式高风险表达；
3. 将持续困扰、睡眠/学习影响等功能受损信号纳入 medium 判断；
4. 区分用户本人风险表达与论文、研究、宣传、程序等非求助语境；
5. 保持确定性、可解释、可测试。
"""
from __future__ import annotations

from core.schemas import RiskAssessment, RiskLevel


class RiskAnalyzer:
    # 明确的高风险表达。
    _high_signals = (
        "撑不下去",
        "撑不住",
        "不想活",
        "想死",
        "自杀",
        "伤害自己",
        "结束生命",
        "活不下去",
    )

    # 隐式但仍需要高度关注的表达。
    # 这里不是简单把“消失”单独作为高风险词，
    # 而是要求它和个人语境共同出现。
    _indirect_high_signals = (
        "从这个世界消失",
        "彻底消失",
        "结束这一切",
        "再也不醒来",
        "不用再醒来",
    )

    # 较明确的中风险困扰信号。
    _medium_signals = (
        "焦虑",
        "睡不着",
        "失眠",
        "很难受",
        "特别难受",
        "压力很大",
        "崩溃",
        "绝望",
        "连续几天睡不好",
        "整晚睡不好",
        "睡不好",
        "压抑",
        "胸口堵",
    )

    # 单独出现时不一定足以判 medium，
    # 但多个同时出现通常意味着明显困扰或功能影响。
    _functional_signals = (
        "状态很差",
        "白天也没精神",
        "没精神",
        "提不起劲",
        "影响到上课",
        "影响上课",
        "影响学习",
        "影响工作",
        "影响睡眠",
        "不太想一个人扛",
        "一个人扛着",
    )

    # 这些词说明风险相关表达可能只是“讨论对象”，
    # 而不是用户本人正在表达风险。
    _meta_context_markers = (
        "论文",
        "论文题目",
        "研究",
        "研究框架",
        "讲座",
        "宣传周",
        "宣传活动",
        "活动报名",
        "程序",
        "代码",
        "关键词",
        "检测",
        "翻译",
        "小说",
        "课程作业",
        "职业",
    )

    # 与“个人正在经历困扰”关系更强的表达。
    # 故意不使用单独的“我”或“自己”，
    # 因为“我在做一个检测伤害自己关键词的程序”属于反例。
    _personal_distress_markers = (
        "我最近",
        "我这几天",
        "我一直",
        "我现在",
        "我开始觉得",
        "我真的",
        "我已经",
        "最近我",
        "这几天我",
        "让我很",
        "已经影响到",
        "已经影响",
        "我不想活",
        "我想死",
        "我撑不住",
        "我撑不下去",
        "我活不下去",
    )

    _relief_markers = (
        "更轻松",
        "能解脱",
        "解脱",
        "就好了",
        "会更好",
    )

    def analyze(self, message: str) -> RiskAssessment:
        text = message.strip().lower()

        # 1) 先判断是否明显是在讨论论文、研究、程序、宣传等对象。
        #
        # 但不能只要出现 meta context 就直接降级：
        #
        # “我在写论文，但我最近真的不想活了”
        #
        # 这种同时包含真实个人风险表达的消息仍然必须继续进入风险检测。
        meta_context = self._is_meta_context(text)
        personal_distress = self._has_personal_distress(text)

        if meta_context and not personal_distress:
            return RiskAssessment(
                risk_level=RiskLevel.LOW,
                signals=["contextual_non_personal_reference"],
                confidence=0.88,
            )

        # 2) 明确 high-risk 表达。
        high_hits = [
            signal
            for signal in self._high_signals
            if signal in text
        ]

        if high_hits:
            return RiskAssessment(
                risk_level=RiskLevel.HIGH,
                signals=high_hits,
                confidence=min(
                    0.90 + 0.02 * (len(high_hits) - 1),
                    0.98,
                ),
            )

        # 3) 隐式 high-risk。
        #
        # “消失”本身不能算 high：
        # “文件怎么突然消失了”显然不是风险表达。
        #
        # 因此要求：
        #   隐式风险表达
        #   +
        #   个人困扰语境
        #
        # 如果同时出现“解脱 / 更轻松”等结果期待，则进一步增强证据。
        indirect_high_hits = [
            signal
            for signal in self._indirect_high_signals
            if signal in text
        ]

        if indirect_high_hits and personal_distress:
            relief_hits = [
                marker
                for marker in self._relief_markers
                if marker in text
            ]

            signals = indirect_high_hits + relief_hits

            return RiskAssessment(
                risk_level=RiskLevel.HIGH,
                signals=signals,
                confidence=0.92 if relief_hits else 0.88,
            )

        # 4) 明确 medium-risk 困扰。
        medium_hits = [
            signal
            for signal in self._medium_signals
            if signal in text
        ]

        if medium_hits:
            return RiskAssessment(
                risk_level=RiskLevel.MEDIUM,
                signals=medium_hits,
                confidence=min(
                    0.78 + 0.03 * (len(medium_hits) - 1),
                    0.92,
                ),
            )

        # 5) 功能影响属于较弱证据。
        #
        # 单独一个“状态很差”比较模糊，因此要求至少两个弱信号，
        # 或者一个明确的“已经影响...”信号。
        functional_hits = [
            signal
            for signal in self._functional_signals
            if signal in text
        ]

        has_explicit_impairment = any(
            marker in text
            for marker in (
                "影响到上课",
                "影响上课",
                "影响学习",
                "影响工作",
                "影响睡眠",
            )
        )

        if len(functional_hits) >= 2 or has_explicit_impairment:
            return RiskAssessment(
                risk_level=RiskLevel.MEDIUM,
                signals=functional_hits,
                confidence=min(
                    0.72 + 0.04 * len(functional_hits),
                    0.88,
                ),
            )

        return RiskAssessment(
            risk_level=RiskLevel.LOW,
            signals=[],
            confidence=0.75,
        )

    def _is_meta_context(self, text: str) -> bool:
        return any(
            marker in text
            for marker in self._meta_context_markers
        )

    def _has_personal_distress(self, text: str) -> bool:
        return any(
            marker in text
            for marker in self._personal_distress_markers
        )