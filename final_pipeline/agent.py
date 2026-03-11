import os
import sys
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from colorama import init, Fore, Style

from data_pipeline.storage import query_hybrid

init(autoreset=True)

def format_docs(docs):
    """Formats retrieved documents into a single context string."""
    formatted = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "Unknown")
        header = doc.metadata.get("header", "N/A")
        cat = doc.metadata.get("category", "Unknown")
        content = doc.page_content
        formatted.append(f"[Document {i}]\nSource: {source}\nHeader: {header}\nCategory: {cat}\nContent:\n{content}\n")
    return "\n".join(formatted)

def get_rag_chain():
    load_dotenv()
    
    # Initialize Model
    llm = ChatOpenAI(model="gpt-5-mini", temperature=0)
    
    # Define RAG Prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are a helpful expert assistant.\n"
         "Use the following pieces of retrieved context to answer the user's question.\n"
         "If you don't know the answer or the context doesn't contain the answer, just say that you don't know.\n"
         "Provide clear, code-centric answers where applicable.\n"
         "Always answer in Korean.\n\n"
         "Context:\n{context}"),
        ("human", "{question}")
    ])
    
    # Create the simple RAG Chain (Prompt + LLM + String Output)
    return prompt | llm | StrOutputParser()

def ask_query(question: str, category: str = None) -> str:
    """
    RAG 파이프라인에 질문을 던지고, 생성된 답변을 문자열로 반환합니다.
    (메인 코드 에이전트 등 외부 모듈에서 Import하여 사용하기 위한 용도)
    """
    rag_chain = get_rag_chain()
    
    # 1. 문서 검색 (Retrieve Documents) - Reranker 활성화 (최종 파이프라인과 동일한 스펙)
    retrieved_docs = query_hybrid(question, k=5, category=category, use_reranker=True)
    context_str = format_docs(retrieved_docs)
    
    # 2. 결과 생성 및 텍스트 반환
    answer = rag_chain.invoke({"context": context_str, "question": question})
    return answer

def run_simple_rag():
    print(f"{Fore.CYAN}=== Final Pipeline Simple RAG Chatbot (Type 'exit' to quit) ==={Style.RESET_ALL}")
    
    while True:
        try:
            user_input = input(f"\n{Fore.GREEN}User: {Style.RESET_ALL}").strip()
            if user_input.lower() in ["exit", "quit", "q"]:
                print("Goodbye!")
                break
            if not user_input:
                continue
                
            print(f"{Fore.MAGENTA}[Retrieving context & generating answer...]{Style.RESET_ALL}", end="\r")
            
            # 함수를 호출하여 문자열(String) 결과를 그대로 받아옴
            answer = ask_query(user_input)
            
            print(f"{Fore.MAGENTA}[Done]                                         {Style.RESET_ALL}")
            print(f"\n{Fore.YELLOW}Agent:{Style.RESET_ALL}\n{answer}\n")
            
        except KeyboardInterrupt:
            print("\nInterrupted.")
            break
        except Exception as e:
            print(f"\nError: {e}")

if __name__ == "__main__":
    run_simple_rag()
