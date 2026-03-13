from typing import TypedDict, Optional
from langchain_core.documents import Document


class AgentState(TypedDict):
    """LangGraph 전체 플로우에서 공유되는 상태"""
    question: str                       # 사용자 원본 질문
    rewritten_query: Optional[str]      # 재작성된 쿼리 (None이면 원본 질문 사용)
    category: Optional[str]             # 추론된 카테고리 (None이면 전체 검색)
    should_rewrite: bool                # 쿼리 재작성 필요 여부
    documents: list[Document]           # 검색된 문서 목록
    answer: str                         # 최종 답변
