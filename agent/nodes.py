import sys
import os
from typing import Any, Optional

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

# 프로젝트 루트 경로를 sys.path에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_pipeline.storage import query_hybrid
from agent.state import AgentState
from agent.prompts import (
    ANALYZE_PROMPT,
    REWRITE_PROMPT,
    GENERATE_PROMPT,
    SUPPORTED_CATEGORIES,
)


# ──────────────────────────────────────────────
# Structured Output 스키마 정의
# ──────────────────────────────────────────────
class AnalyzeOutput(BaseModel):
    should_rewrite: bool = Field(description="Whether the query needs to be rewritten for better search results")
    category: Optional[str] = Field(default=None, description="The most relevant Spring documentation category, or null")
    reason: str = Field(description="Brief reason for the decision")


class RewriteOutput(BaseModel):
    rewritten_query: str = Field(description="Rewritten search query optimized for semantic search")


class GenerateOutput(BaseModel):
    answer: str = Field(description="Final answer to the user's question in Korean")


def _get_llm(temperature: float = 0) -> ChatOpenAI:
    return ChatOpenAI(model="gpt-5-mini", temperature=temperature)


def _format_docs(docs: list) -> str:
    """검색 결과 문서를 컨텍스트 문자열로 포맷팅"""
    formatted = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "Unknown")
        header = doc.metadata.get("header", "N/A")
        category = doc.metadata.get("category", "Unknown")
        formatted.append(
            f"[Document {i}]\n"
            f"Source: {source}\n"
            f"Header: {header}\n"
            f"Category: {category}\n"
            f"Content:\n{doc.page_content}\n"
        )
    return "\n".join(formatted)


# ──────────────────────────────────────────────
# Node 1: Analyze
#   - 쿼리 재작성 필요 여부 판단
#   - 카테고리 자동 추론
# ──────────────────────────────────────────────
def analyze_node(state: AgentState) -> dict[str, Any]:
    llm = _get_llm().with_structured_output(AnalyzeOutput)
    chain = ANALYZE_PROMPT | llm

    categories_str = "\n".join(f"  - {c}" for c in SUPPORTED_CATEGORIES)
    result: AnalyzeOutput = chain.invoke({
        "question": state["question"],
        "categories": categories_str,
    })

    category = result.category
    if category not in SUPPORTED_CATEGORIES:
        category = None

    return {
        "should_rewrite": result.should_rewrite,
        "category": category,
    }


# ──────────────────────────────────────────────
# Node 2: Rewrite
#   - 검색에 최적화된 쿼리로 변환
# ──────────────────────────────────────────────
def rewrite_node(state: AgentState) -> dict[str, Any]:
    llm = _get_llm().with_structured_output(RewriteOutput)
    chain = REWRITE_PROMPT | llm

    result: RewriteOutput = chain.invoke({
        "question": state["question"],
        "category": state.get("category") or "null",
    })

    return {"rewritten_query": result.rewritten_query}


# ──────────────────────────────────────────────
# Node 3: Retrieve
#   - Hybrid(BM25 + Chroma) + Reranker 검색
# ──────────────────────────────────────────────
def retrieve_node(state: AgentState) -> dict[str, Any]:
    query = state.get("rewritten_query") or state["question"]
    category = state.get("category")

    docs = query_hybrid(
        query=query,
        k=5,
        category=category,
        use_reranker=True,
    )

    return {"documents": docs}


# ──────────────────────────────────────────────
# Node 4: Generate
#   - 검색 결과를 바탕으로 최종 답변 생성
# ──────────────────────────────────────────────
def generate_node(state: AgentState) -> dict[str, Any]:
    llm = _get_llm().with_structured_output(GenerateOutput)
    chain = GENERATE_PROMPT | llm

    context = _format_docs(state["documents"])
    result: GenerateOutput = chain.invoke({
        "question": state["question"],
        "context": context,
    })

    return {"answer": result.answer}
