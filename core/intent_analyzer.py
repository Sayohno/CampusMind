"""CampusMind IntentAnalyzer：加权意图信号 + 目标导向打分 + 最小必要上下文继承。

当前消息优先使用可解释的确定性信号识别业务目标；只有当前消息缺少明确
业务信号且明显是 follow-up 时，才从 CaseState.topic 或最近用户消息继承意图。

设计目标：
1. 不再把所有关键词都视为同等重要；
2. 区分“句子里提到了什么”与“用户当前想做什么”；
3. 保留确定性、可测试和可解释性；
4. 不改变既有 follow-up 上下文继承边界。
"""
from __future__ import annotations

from core.schemas import ConversationMessage, IntentResult, IntentType


class IntentAnalyzer:
    _signals: dict[IntentType, dict[str, float]] = {
        IntentType.RESOURCE_QUERY: {
            "咨询中心": 2.0,
            "预约": 2.6,
            "地址": 2.2,
            "开放时间": 2.6,
            "几点": 1.8,
            "电话": 2.2,
            "联系方式": 2.6,
            "在哪里": 2.0,
            "热线": 2.0,
            "资源": 2.2,
            "报名": 1.8,
            "申请": 1.4,
            "讲座": 1.0,
            "通知": 1.0,
        },
        IntentType.HUMAN_SUPPORT: {
            "找个人": 3.0,
            "真人": 3.0,
            "联系到人": 3.0,
            "人工帮助": 3.0,
            "找人帮忙": 3.0,
            "真人帮忙": 3.0,
            "心理咨询师": 0.8,
            "专业老师": 0.8,
        },
        IntentType.ACADEMIC_CAREER: {
            "秋招": 3.0,
            "简历": 3.0,
            "求职": 3.0,
            "面试": 3.0,
            "实习": 3.0,
            "学业": 2.5,
            "考试": 2.5,
            "论文": 2.5,
            "就业": 2.2,
            "考研": 3.0,
            "找工作": 3.0,
            "投递": 2.6,
            "职业": 2.0,
            "项目经历": 2.2,
            "课程": 1.8,
            "作业": 1.8,
            "实验": 1.6,
            "复习": 1.8,
            "毕业": 1.2,
        },
        IntentType.RELATIONSHIP: {
            "室友": 3.0,
            "同学": 1.8,
            "朋友": 1.6,
            "恋爱": 2.5,
            "失恋": 3.0,
            "分手": 3.0,
            "人际": 2.3,
            "关系": 1.8,
            "矛盾": 2.4,
            "冲突": 2.2,
            "闹翻": 2.5,
            "对方": 0.8,
            "那个人": 0.8,
            "很僵": 1.4,
            "越来越僵": 1.8,
        },
        IntentType.EMOTIONAL_SUPPORT: {
            "难受": 2.2,
            "焦虑": 2.2,
            "压力": 1.6,
            "睡不好": 2.2,
            "睡不着": 2.2,
            "失眠": 2.2,
            "孤独": 2.2,
            "想聊聊": 1.5,
            "情绪": 2.2,
            "心情": 1.8,
            "低落": 2.2,
            "崩溃": 2.2,
            "绝望": 2.2,
            "提不起劲": 2.4,
            "发呆": 1.7,
            "压抑": 2.2,
            "胸口堵": 2.2,
            "状态很差": 1.8,
        },
        IntentType.GENERAL: {
            "你能做什么": 4.0,
            "介绍一下你": 3.0,
            "翻译": 2.4,
            "润色": 2.2,
            "程序": 1.8,
            "代码": 1.8,
        },
    }

    _resource_goal_markers = (
        "怎么预约",
        "如何预约",
        "怎么报名",
        "在哪里报名",
        "哪里报名",
        "怎么申请",
        "从哪里申请",
        "在哪里",
        "几点",
        "开放时间",
        "电话",
        "联系方式",
        "有哪些资源",
        "什么资源",
        "哪里看通知",
    )

    _career_goal_markers = (
        "怎么准备",
        "如何准备",
        "怎么规划",
        "如何规划",
        "怎么安排",
        "如何安排",
        "怎么复习",
        "就业方向",
        "职业方向",
        "找工作",
        "投递",
        "项目经历",
        "突出自己的经历",
        "补哪些能力",
    )

    _human_goal_markers = (
        "找个人",
        "找个真人",
        "真人帮忙",
        "找人帮忙",
        "联系一个真人",
        "联系到人",
        "人工帮助",
    )

    _relationship_goal_markers = (
        "怎么沟通",
        "怎么开口",
        "怎么聊",
        "和对方说",
    )

    _tie_priority = (
        IntentType.RESOURCE_QUERY,
        IntentType.HUMAN_SUPPORT,
        IntentType.ACADEMIC_CAREER,
        IntentType.RELATIONSHIP,
        IntentType.EMOTIONAL_SUPPORT,
        IntentType.GENERAL,
    )

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

        # 1) 显式新意图永远优先。
        current_intent, current_score = self._best_intent(text)
        if current_score > 0:
            return IntentResult(
                intent=current_intent,
                confidence=self._confidence(current_score),
            )

        # 2) 当前句缺少业务信号且明显是 follow-up 时，才继承上下文。
        if self.is_follow_up(message):
            inherited_from_case = self._intent_from_case_topic(case_topic)
            if inherited_from_case is not None:
                return IntentResult(intent=inherited_from_case, confidence=0.80)

            inherited_from_memory = self._intent_from_recent_user_memory(
                memory_history or []
            )
            if inherited_from_memory is not None:
                return IntentResult(
                    intent=inherited_from_memory,
                    confidence=0.74,
                )

        # 3) 没有可靠依据时保持 General，不强猜。
        return IntentResult(
            intent=IntentType.GENERAL,
            confidence=0.55,
        )

    def is_follow_up(self, message: str) -> bool:
        text = message.strip().lower()
        return any(marker in text for marker in self._follow_up_markers)

    def _best_intent(self, text: str) -> tuple[IntentType, float]:
        scores = {
            intent: 0.0
            for intent in self._signals
        }

        # 第一层：主题/语义信号。
        # 不同词具有不同区分能力，因此使用不同权重。
        for intent, signals in self._signals.items():
            for phrase, weight in signals.items():
                if phrase in text:
                    scores[intent] += weight

        # 第二层：用户当前目标。
        # “在哪里 / 怎么报名”比单纯出现某个主题词更能体现 Resource Query。
        if any(marker in text for marker in self._resource_goal_markers):
            scores[IntentType.RESOURCE_QUERY] += 2.0

        # “有没有 + 校园资源实体”也是典型资源可用性查询。
        if (
            "有没有" in text
            and any(
                noun in text
                for noun in (
                    "讲座",
                    "活动",
                    "资源",
                    "中心",
                    "热线",
                    "通知",
                )
            )
        ):
            scores[IntentType.RESOURCE_QUERY] += 2.0

        # 求职/学业类目标动作。
        if any(marker in text for marker in self._career_goal_markers):
            scores[IntentType.ACADEMIC_CAREER] += 1.8

        # 明确希望现实中的人介入。
        # 基础词本身已经有较高权重，所以这里只做轻量 bonus。
        if any(marker in text for marker in self._human_goal_markers):
            scores[IntentType.HUMAN_SUPPORT] += 0.2

        # 明确询问关系沟通方法。
        if any(marker in text for marker in self._relationship_goal_markers):
            scores[IntentType.RELATIONSHIP] += 1.5

        # 明确 tie-break，避免结果暗中依赖 dict 的声明顺序。
        best_intent = max(
            self._tie_priority,
            key=lambda intent: scores[intent],
        )
        return best_intent, scores[best_intent]

    @staticmethod
    def _confidence(score: float) -> float:
        return min(
            0.66 + 0.06 * score,
            0.94,
        )

    @staticmethod
    def _intent_from_case_topic(
        case_topic: str | None,
    ) -> IntentType | None:
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
        # 只使用用户历史，不让 assistant 生成文本反向污染路由。
        for item in reversed(memory_history):
            if item.role != "user":
                continue

            intent, score = self._best_intent(
                item.content.strip().lower()
            )
            if score > 0:
                return intent

        return None