"""SupportCaseState 持久化抽象与进程内实现。"""
from __future__ import annotations

from abc import ABC, abstractmethod

from core.schemas import CaseStatus, RiskLevel, SupportCaseState, utc_now


class CaseStore(ABC):
    @abstractmethod
    def get(self, *, user_id: str, conversation_id: str) -> SupportCaseState | None:
        raise NotImplementedError

    @abstractmethod
    def save(self, state: SupportCaseState) -> SupportCaseState:
        raise NotImplementedError

    def ensure(
        self,
        *,
        user_id: str,
        conversation_id: str,
        topic: str,
        risk_level: RiskLevel,
    ) -> SupportCaseState:
        current = self.get(user_id=user_id, conversation_id=conversation_id)
        if current is None:
            current = SupportCaseState(
                case_id=f"case:{user_id}:{conversation_id}",
                user_id=user_id,
                conversation_id=conversation_id,
                topic=topic,
                current_risk_level=risk_level,
            )
        else:
            current.topic = topic
            current.current_risk_level = risk_level
            current.updated_at = utc_now()
        return self.save(current)


class InMemoryCaseStore(CaseStore):
    def __init__(self) -> None:
        self._states: dict[tuple[str, str], SupportCaseState] = {}

    def get(self, *, user_id: str, conversation_id: str) -> SupportCaseState | None:
        state = self._states.get((user_id, conversation_id))
        return None if state is None else state.model_copy(deep=True)

    def save(self, state: SupportCaseState) -> SupportCaseState:
        state.updated_at = utc_now()
        self._states[(state.user_id, state.conversation_id)] = state.model_copy(deep=True)
        return state.model_copy(deep=True)

    def update_status(
        self,
        *,
        user_id: str,
        conversation_id: str,
        status: CaseStatus,
    ) -> SupportCaseState:
        state = self.get(user_id=user_id, conversation_id=conversation_id)
        if state is None:
            raise KeyError("SupportCaseState 不存在")
        state.status = status
        return self.save(state)
