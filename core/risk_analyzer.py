"""CampusMind RiskAnalyzer：风险信号、语境与功能影响的确定性风险识别。

这里只进行系统内部工程风险分级，不进行医疗诊断。

RiskAnalyzer 本身仍然是确定性规则模块。

当规则检测到：
- 潜在 high-risk 但上下文存在歧义；
- 风险词出现在论文 / 小说 / 程序 / 活动等 meta context；
- 存在功能影响但固定风险词不足；

可以通过 needs_semantic_fallback() 告诉上层：
当前结果值得再交给 SemanticAnalyzer 做语义消歧。

最终 Agent、mandatory tools 和 escalation policy
仍然由 RiskPolicyEngine 决定。
"""
from __future__ import annotations

from core.schemas import RiskAssessment, RiskLevel


class RiskAnalyzer:
    # 明确高风险表达。
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

    # 隐式但需要高度关注的表达。
    #
    # 注意：
    # 不能把单独的“消失”直接视为 high，
    # 因为“文件消失了”完全不是风险表达。
    _indirect_high_signals = (
        "从这个世界消失",
        "彻底消失",
        "结束这一切",
        "再也不醒来",
        "不用再醒来",
    )

    # 中等级明显困扰信号。
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

    # 功能影响类信号。
    #
    # 单独出现时不一定直接代表 medium，
    # 但多个一起出现，或者明确影响学习 / 工作 / 睡眠，
    # 就值得提高风险等级。
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

    # 风险相关词可能只是被讨论的对象，
    # 而不是用户本人正在表达风险。
    #
    # 例如：
    # “自杀预防宣传周在哪里报名？”
    # “小说人物说‘我不想活了’，怎么翻译？”
    # “检测‘伤害自己’关键词的程序怎么写？”
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
        "文章",
        "润色",
        "讨论",
        "课程",
        "题目",
    )

    # 较强的“用户本人正在经历困扰”表达。
    #
    # 不使用单独的“我”或“自己”做判断，
    # 否则：
    #
    # “我在写检测‘伤害自己’关键词的程序”
    #
    # 会被错误识别成个人风险表达。
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

    # 某些隐式 high-risk 表达会同时描述：
    #
    # “如果消失了会不会更轻松 / 解脱”
    #
    # 这些词不是独立风险词，
    # 只有和 indirect high signal 组合才有意义。
    _relief_markers = (
        "更轻松",
        "能解脱",
        "解脱",
        "就好了",
        "会更好",
    )

    def analyze(
        self,
        message: str,
    ) -> RiskAssessment:
        text = message.strip().lower()

        # 1) 判断是否存在明显 meta context。
        #
        # 如果只是在论文 / 小说 / 程序 / 宣传等语境中讨论风险词，
        # 并且没有明显个人困扰证据，
        # 确定性规则暂时按 LOW 处理。
        #
        # 后续 needs_semantic_fallback() 会判断是否还需要 LLM 消歧。
        meta_context = self._is_meta_context(text)

        personal_distress = (
            self._has_personal_distress(text)
        )

        if (
            meta_context
            and not personal_distress
        ):
            return RiskAssessment(
                risk_level=RiskLevel.LOW,
                signals=[
                    "contextual_non_personal_reference"
                ],
                confidence=0.88,
            )

        # 2) 明确 high-risk。
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
                    0.90
                    + 0.02 * (len(high_hits) - 1),
                    0.98,
                ),
            )

        # 3) 隐式 high-risk。
        #
        # 需要：
        #
        # indirect high expression
        # +
        # personal distress context
        #
        # 才判 high。
        indirect_high_hits = [
            signal
            for signal in self._indirect_high_signals
            if signal in text
        ]

        if (
            indirect_high_hits
            and personal_distress
        ):
            relief_hits = [
                marker
                for marker in self._relief_markers
                if marker in text
            ]

            signals = (
                indirect_high_hits
                + relief_hits
            )

            return RiskAssessment(
                risk_level=RiskLevel.HIGH,
                signals=signals,
                confidence=(
                    0.92
                    if relief_hits
                    else 0.88
                ),
            )

        # 4) 明确 medium-risk。
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
                    0.78
                    + 0.03 * (len(medium_hits) - 1),
                    0.92,
                ),
            )

        # 5) 功能影响。
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

        # 两个及以上弱功能信号，
        # 或一个明确“已经影响...”信号，
        # 视为 medium。
        if (
            len(functional_hits) >= 2
            or has_explicit_impairment
        ):
            return RiskAssessment(
                risk_level=RiskLevel.MEDIUM,
                signals=functional_hits,
                confidence=min(
                    0.72
                    + 0.04 * len(functional_hits),
                    0.88,
                ),
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
        """判断当前 Risk 规则结果是否需要进一步做语义消歧。

        True 不代表确定性规则一定错误。

        它只表示：
        当前文本存在明显语境冲突、隐式风险或功能影响，
        值得交给 SemanticAnalyzer 再判断一次。
        """
        text = message.strip().lower()

        high_hits = [
            signal
            for signal in self._high_signals
            if signal in text
        ]

        indirect_high_hits = [
            signal
            for signal in self._indirect_high_signals
            if signal in text
        ]

        medium_hits = [
            signal
            for signal in self._medium_signals
            if signal in text
        ]

        # 风险词与论文 / 小说 / 程序 / 活动等
        # meta context 同时存在。
        #
        # 典型：
        # “小说人物说‘我不想活了’，怎么翻译？”
        #
        # 这种 hard negative 很适合交给语义模型消歧。
        if (
            self._is_meta_context(text)
            and (
                high_hits
                or indirect_high_hits
                or medium_hits
            )
        ):
            return True

        # 规则最终给 LOW，
        # 但文本里其实存在潜在 high-risk phrase。
        #
        # 有可能是：
        # - 真正的隐式高风险；
        # - 也可能只是引用 / 研究语境。
        #
        # 都应该进一步消歧。
        if (
            result.risk_level == RiskLevel.LOW
            and (
                high_hits
                or indirect_high_hits
            )
        ):
            return True

        functional_hits = [
            signal
            for signal in self._functional_signals
            if signal in text
        ]

        # 持续性 / 个人体验线索。
        temporal_or_personal = any(
            marker in text
            for marker in (
                "最近",
                "这几天",
                "我觉得",
                "我总",
                "我一直",
                "我现在",
                "第二天",
                "越来越",
            )
        )

        # 例如：
        #
        # “最近晚上脑子停不下来，
        # 第二天上课也没精神。”
        #
        # 固定 medium phrase 可能没有命中，
        # 但已经存在持续性 + 功能影响。
        if (
            result.risk_level == RiskLevel.LOW
            and functional_hits
            and temporal_or_personal
        ):
            return True

        return False

    def has_potential_high_signal(
        self,
        message: str,
    ) -> bool:
        """判断消息中是否存在潜在 high-risk signal。

        这个方法不是最终风险分类器。

        它主要用于：
        Semantic fallback 本来应该执行，
        但模型调用 / JSON 解析失败时，
        上层可以选择安全侧 fail-closed。
        """
        text = message.strip().lower()

        return any(
            signal in text
            for signal in (
                *self._high_signals,
                *self._indirect_high_signals,
            )
        )

    def _is_meta_context(
        self,
        text: str,
    ) -> bool:
        return any(
            marker in text
            for marker in self._meta_context_markers
        )

    def _has_personal_distress(
        self,
        text: str,
    ) -> bool:
        return any(
            marker in text
            for marker in self._personal_distress_markers
        )