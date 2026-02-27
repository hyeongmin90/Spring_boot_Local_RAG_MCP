import os
import sys
from dotenv import load_dotenv

# 부모 디렉토리를 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from langsmith import Client
from langsmith.evaluation import evaluate
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from data_pipeline.storage import query_documents
from openevals.prompts import CORRECTNESS_PROMPT, RAG_GROUNDEDNESS_PROMPT, RAG_RETRIEVAL_RELEVANCE_PROMPT
from openevals.llm import create_llm_as_judge

load_dotenv()

# 평가 심사위원(Judge)으로 사용할 LLM 설정
judge_llm = ChatOpenAI(model="gpt-5-mini", temperature=0)

from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage

from data_pipeline.rag_agent import search_spring_boot_docs

# ==========================================
# 0. 실제 RAG 에이전트 초기화
# ==========================================
def initialize_agent():
    llm = ChatOpenAI(model="gpt-5-mini", temperature=0)
    tools = [search_spring_boot_docs]
    system_prompt = (
        "You are a Spring Boot Expert RAG Agent.\n"
        "Answer user questions accurately using the provided documentation.\n"
        "ALWAYS use the 'search_spring_boot_docs' tool to verify information before answering.\n"
        "For the first search, enter the user's question as is without specifying a document type.\n"
        "If you cannot find the answer in the search results, admit it honestly.\n"
        "Do not include any information not found in the search results.\n"
        "If search results are insufficient, retry with different keywords before giving up.\n"
        "Provide clear, code-centric answers where applicable.\n"
        "Do not use search_spring_boot_docs tool more than 4 times.\n"
        ""
        # "Answer in Korean."
    )
    
    agent = create_agent(
        model=llm,
        tools=tools,
        checkpointer=InMemorySaver(),
        system_prompt=system_prompt,
    )
    return agent

# 글로벌로 한 번만 생성해 둡니다. (테스트마다 매번 생성하면 느려짐)
eval_agent = initialize_agent()

# ==========================================
# 1. 평가할 대상 함수 (Target Function) 정의
# ==========================================
def predict_rag(inputs: dict) -> dict:
    """
    명시된 질문에 대해 실제 rag_agent 파이프라인(LangGraph)을 가동하여 답변과 검색된 컨텍스트를 반환합니다.
    """
    question = inputs["question"]
    
    # 세션 관리를 위해 임의의 thread_id 생성 (평가 건마다 독립적인 스레드)
    import uuid
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    
    # Agent 실행
    result = eval_agent.invoke({"messages": [HumanMessage(content=question)]}, config)
    messages = result["messages"]
    
    # 최종 답변과 검색된 컨텍스트 추출
    final_answer = ""
    accumulated_context = ""
    
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.content:
            final_answer = msg.content
        elif isinstance(msg, ToolMessage):
            # 검색 도구가 반환한 문자열(context)을 수집
            accumulated_context += msg.content + "\n\n"
            
    return {
        "prediction": final_answer,
        "context": accumulated_context.strip() # context_qa 평가 시 환각 여부를 판단하기 위해 넘김
    }



qa_evaluator = create_llm_as_judge(
    prompt=CORRECTNESS_PROMPT,
    judge=judge_llm,
    feedback_key="qa",
    continuous=True,     # 0.0 ~ 1.0 점수
    use_reasoning=True   # 판단에 대한 이유 포함
)

context_evaluator = create_llm_as_judge(
    prompt=RAG_GROUNDEDNESS_PROMPT,
    judge=judge_llm,
    feedback_key="context_qa",
    continuous=True,
    use_reasoning=True
)

retrieval_evaluator = create_llm_as_judge(
    prompt=RAG_RETRIEVAL_RELEVANCE_PROMPT,
    feedback_key="retrieval_relevance",
    judge=judge_llm,
    continuous=True,
    use_reasoning=True
)

def run_qa_eval(run, example):
    """
    LangSmith evaluate()에서 qa_evaluator를 호출하기 위해 감싸는 래퍼 함수입니다.
    """
    prediction = run.outputs["prediction"]
    reference = example.outputs["answer"]
    question = example.inputs["question"]
    
    # CORRECTNESS_PROMPT는 {inputs}, {reference_outputs}, {outputs}를 요구하므로 맞춰서 넘깁니다.
    result = qa_evaluator(
        inputs=question,
        reference_outputs=reference,
        outputs=prediction
    )
    return result

def run_context_eval(run, example):
    """
    LangSmith evaluate()에서 context_evaluator를 호출하기 위해 감싸는 래퍼 함수입니다.
    """
    prediction = run.outputs["prediction"]
    context = run.outputs["context"]
    
    # RAG_GROUNDEDNESS_PROMPT는 {context}, {outputs}를 요구하므로 맞춰서 넘깁니다.
    result = context_evaluator(
        context=context,
        outputs=prediction
    )
    return result

def run_retrieval_eval(run, example):
    """
    LangSmith evaluate()에서 retrieval_evaluator를 호출하기 위해 감싸는 래퍼 함수입니다.
    """
    question = example.inputs["question"]
    context = run.outputs["context"]
    
    # RAG_RETRIEVAL_RELEVANCE_PROMPT {context}, {inputs}
    result = retrieval_evaluator(
        context=context,
        inputs=question
    )
    return result

def run_evaluation():
    dataset_name="sampled_50_questions"
    print(f"=== '{dataset_name}' 데이터셋을 이용한 RAG 파이프라인 평가 시작 ===")
    
    experiment_results = evaluate(
        predict_rag,
        data=dataset_name,
        evaluators=[run_qa_eval, run_context_eval, run_retrieval_eval],
        experiment_prefix="RAG-eval", 
        description="Spring Boot RAG 파이프라인의 정확도(qa) 및 문서 기반 답변 생성(context_qa) 여부 통합 테스트"
    )
    
    print("\n평가가 완료되었습니다. 출력된 LangSmith 대시보드 URL을 클릭하여 결과를 확인하세요.")

if __name__ == "__main__":
    run_evaluation()
