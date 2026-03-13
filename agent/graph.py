from langgraph.graph import StateGraph, START, END

from agent.state import AgentState
from agent.nodes import analyze_node, rewrite_node, retrieve_node, generate_node


def _should_rewrite_router(state: AgentState) -> str:
    """analyze 노드 이후 분기: 재작성 필요 여부에 따라 라우팅"""
    if state.get("should_rewrite"):
        return "rewrite"
    return "retrieve"


def build_graph():
    """LangGraph StateGraph를 조립하여 컴파일된 그래프를 반환"""
    graph = StateGraph(AgentState)

    # 노드 등록
    graph.add_node("analyze", analyze_node)
    graph.add_node("rewrite", rewrite_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)

    # 엣지 연결
    graph.add_edge(START, "analyze")
    graph.add_conditional_edges(
        "analyze",
        _should_rewrite_router,
        {
            "rewrite": "rewrite",
            "retrieve": "retrieve",
        },
    )
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    return graph.compile()
