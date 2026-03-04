import os
import sys
import random
from tqdm import tqdm
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List
from datetime import datetime

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from data_pipeline.storage import get_vectorstore, query_documents, mmr_query_documents, query_hybrid

# Load environment variables (OPENAI_API_KEY)
load_dotenv()

def get_random_chunks(n=10):
    """
    Fetches random chunks from the vector store.
    """
    vectorstore = get_vectorstore()
    
    # Get all documents from Chroma
    # By default, vectorstore.get() returns all documents if no filter is applied
    print("Fetching documents from vector store...")
    db_data = vectorstore.get()
    
    if not db_data or 'documents' not in db_data or not db_data['documents']:
        print("No documents found in the vector store.")
        return []
        
    total_docs = len(db_data['documents'])
    print(f"Total documents available: {total_docs}")
    
    # Randomly sample N indices
    sample_size = min(n, total_docs)
    sampled_indices = random.sample(range(total_docs), sample_size)
    
    sampled_chunks = []
    for idx in sampled_indices:
        doc_content = db_data['documents'][idx]
        metadata = db_data['metadatas'][idx] if 'metadatas' in db_data and db_data['metadatas'] else {}
        doc_id = db_data['ids'][idx] if 'ids' in db_data and db_data['ids'] else None
        
        # Only select chunks that have a reasonable length to generate a good question
        if len(doc_content) > 300:
            sampled_chunks.append({
                "content": doc_content,
                "metadata": metadata,
                "id": doc_id
            })
            
    print(f"Sampled {len(sampled_chunks)} viable chunks for evaluation (length > 300).")
    return sampled_chunks

class Questions(BaseModel):
    questions: List[str]

def generate_questions(chunk_content):
    """
    Uses LLM to generate 3 challenging questions based on the provided text chunk.
    """
    llm = ChatOpenAI(model="gpt-5-mini", temperature=0.5)

    prompt = """
    당신은 Retrieval 시스템을 평가하기 위한 데이터셋을 생성하는 전문가입니다.

    웹 문서에서 추출된 하나의 텍스트 청크가 주어집니다.
    이 청크의 내용을 기반으로 최대 1개의 질문을 생성하세요.

    이 데이터셋은 Retriever의 한계를 테스트하고 성능(Recall@k, MRR 등)을 변별력 있게 평가하기 위한 **최상위 난이도(Very Hard)** 목적입니다.
    현재 단순 검색(BM25/Dense) 모델들이 너무 쉽게 정답을 찾고 있으므로, 검색 난이도를 극단적으로 높이는 것이 핵심입니다.

    다음 규칙을 반드시 따르세요.

    [문서 품질 평가]
    1. 텍스트 청크에 의미 있는 정보가 부족하거나, 코드만 존재하거나, 메뉴/네비게이션 정보만 있는 경우에는 질문을 생성하지 마세요.

    [질문 생성 난이도 극대화 규칙 - 필수]
    2. **어휘적 중복 최소화 (Lexical Isolation):** 질문에 텍스트의 핵심 키워드, 고유명사, 메서드명, 클래스명 등을 절대 그대로 사용하지 마세요. 반드시 그 의미를 풀어쓴 개념적 설명이나 완전한 유의어(Synonyms)로 대체하세요.
    3. **추상화 및 일반화 (Abstraction):** 특정 기술이나 도구의 이름을 직접 언급하는 대신, "이러한 역할을 하는 컴포넌트", "해당 프레임워크의 특정 설정" 등으로 추상화하여 질문하세요.
    4. **의미적 재구성 (Semantic Restructuring):** 텍스트의 문장 구조를 완전히 뒤집고, 제공된 정보의 결과를 원인으로 묻거나, 원인을 결과로 묻는 등 역방향 추론이 필요한 질문을 만드세요.
    5. **다중 정보 결합 (Multi-hop Reasoning):** 청크 내의 흩어져 있는 두 가지 이상의 정보를 모두 이해하고 결합해야만 어떤 내용인지 유추할 수 있도록 질문을 복잡하게 구성하세요.
    6. 단순히 "무엇인가?", "어떻게 하는가?" 보다, "어떤 제약이나 문제점 때문에 이러한 아키텍처적 결정을 내리게 되는가?" 와 같은 심층적인 의도를 묻는 질문을 만드세요.

    [Retriever 평가 고려사항]
    7. 질문은 의미적으로는 텍스트와 완벽히 일치하지만, 단순 키워드 매칭(BM25)이나 얕은 의미 검색(Dense)으로는 우연히 찾기 거의 불가능하도록 작성되어야 합니다.
    8. 텍스트에 있는 단어 중 흔하지 않은 단어(Rare words)가 질문에 포함되지 않도록 특별히 주의하세요.

    [출력 형태]
    생성된 질문 문장 1개를 배열에 담아 반환하세요.
    질문은 자연스럽고 전문적인 느낌의 **영어**로 작성하세요.

    {text}
    """
    
    prompt = PromptTemplate.from_template(prompt)
    
    chain = prompt | llm.with_structured_output(Questions)
    questions = chain.invoke({"text": chunk_content})

    return questions.questions

def evaluate_retrieval(question, expected_id, expected_source, method="dense", k=10):
    """
    Queries the vector store with the generated question and checks if the expected chunk is retrieved.
    Calculates the rank of the expected chunk.
    """
    if method == "mmr":
        results = mmr_query_documents(question, k=k)
    elif method == "hybrid":
        results = query_hybrid(question, k=k, use_reranker=False)
    elif method == "hybrid_cohere":
        results = query_hybrid(question, k=k, use_reranker=True)
    else:
        results = query_documents(question, k=k)

    for rank, doc in enumerate(results, start=1):
        retrieved_id = getattr(doc, 'id', None) or doc.metadata.get("chunk_id")
        retrieved_source = doc.metadata.get("source")
        
        # Check if it matches the expected chunk
        # Match only by chunk ID for strict evaluation
        if expected_id and retrieved_id == expected_id:
            return rank
            
    return -1 # Not found within top k

import concurrent.futures

def run_evaluation(num_samples=10, max_k=10):
    print("=== Starting RAG Quantitative Evaluation ===")
    
    chunks = get_random_chunks(n=num_samples)
    if not chunks:
        return
        
    print("\nGenerating questions concurrently...")
    
    # Process question generation concurrently
    def generate_for_chunk(item):
        content = item['content']
        questions = generate_questions(content)
        return item, questions

    chunk_questions_map = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        # Submit all tasks
        future_to_chunk = {executor.submit(generate_for_chunk, item): item for item in chunks}
        
        # Gather results with progress bar
        for future in tqdm(concurrent.futures.as_completed(future_to_chunk), total=len(chunks), desc="LLM Generation"):
            try:
                item, questions = future.result()
                chunk_questions_map.append((item, questions))
            except Exception as exc:
                print(f"\nChunk generation generated an exception: {exc}")

    print("\nEvaluating retrieval...")
    #======================================
    # methods = ["dense", "hybrid", "hybrid_cohere"]
    methods = ["dense", "mmr"]
    #======================================
    all_metrics = {m: {
        "hits_1": 0,
        "hits_5": 0, "mrr_sum_5": 0.0, 
        "hits_10": 0, "mrr_sum_10": 0.0, 
        "hits_max_k": 0, "mrr_sum_max_k": 0.0, 
        "results_log": []
    } for m in methods}
    
    total_questions = 0
    
    for i, (item, questions) in enumerate(tqdm(chunk_questions_map, desc="Retrieval Testing")):
        metadata = item['metadata']
        
        expected_chunk_id = metadata.get("chunk_id")
        expected_source = metadata.get("source", "Unknown")
        
        for q_idx, question in enumerate(questions):
            total_questions += 1
            
            for method in methods:
                # 2. Evaluate Retrieval (Fetch up to max_k, e.g., 50)
                rank = evaluate_retrieval(question, expected_chunk_id, expected_source, method=method, k=max_k)
                
                # 3. Calculate Metrics
                # --- Top 1 Metrics ---
                is_hit_1 = rank == 1
                if is_hit_1:
                    all_metrics[method]["hits_1"] += 1

                # --- Top 5 Metrics ---
                is_hit_5 = 0 < rank <= 5
                if is_hit_5:
                    all_metrics[method]["hits_5"] += 1
                    all_metrics[method]["mrr_sum_5"] += 1.0 / rank

                # --- Top 10 Metrics ---
                is_hit_10 = 0 < rank <= 10
                if is_hit_10:
                    all_metrics[method]["hits_10"] += 1
                    all_metrics[method]["mrr_sum_10"] += 1.0 / rank

                # --- Top max_k Metrics ---
                is_hit_max_k = 0 < rank <= max_k
                if is_hit_max_k:
                    all_metrics[method]["hits_max_k"] += 1
                    all_metrics[method]["mrr_sum_max_k"] += 1.0 / rank
                    
                # Log result for this question
                all_metrics[method]["results_log"].append({
                    "chunk_idx": i + 1,
                    "q_idx": q_idx + 1,
                    "question": question,
                    "expected_source": expected_source,
                    "rank": rank,
                })
        
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    import os
    os.makedirs("results", exist_ok=True)
    unified_log_file = os.path.join("results", f"eval_summary_{timestamp}.txt")
    unified_log_content = f"=== RAG 정량 평가 요약 ({timestamp}) ===\n"
    unified_log_content += f"총 샘플 청크 수: {len(chunks)}\n"
    unified_log_content += f"평가된 총 질문 수: {total_questions}\n"

    for method in methods:
        # Final Metrics calculation
        tq = total_questions if total_questions > 0 else 1
    
        hit_rate_1 = all_metrics[method]["hits_1"] / tq
        
        hit_rate_5 = all_metrics[method]["hits_5"] / tq
        mrr_5 = all_metrics[method]["mrr_sum_5"] / tq
        
        hit_rate_10 = all_metrics[method]["hits_10"] / tq
        mrr_10 = all_metrics[method]["mrr_sum_10"] / tq
    
        hit_rate_max_k = all_metrics[method]["hits_max_k"] / tq
        mrr_max_k = all_metrics[method]["mrr_sum_max_k"] / tq
        
        summary_text = "\n" + "="*50 + "\n"
        summary_text += f"=== Evaluation Results ({method.upper()}) ===\n"
        summary_text += "-" * 25 + "\n"
        summary_text += "--- Top 1 Metrics ---\n"
        summary_text += f"Top-1 Hit Rate: {hit_rate_1:.2%} ({all_metrics[method]['hits_1']}/{total_questions})\n"
        summary_text += "-" * 25 + "\n"
        summary_text += "--- Top 5 Metrics ---\n"
        summary_text += f"Top-5 Hit Rate: {hit_rate_5:.2%} ({all_metrics[method]['hits_5']}/{total_questions})\n"
        summary_text += f"Top-5 MRR: {mrr_5:.4f}\n"
        summary_text += "-" * 25 + "\n"
        summary_text += "--- Top 10 Metrics ---\n"
        summary_text += f"Top-10 Hit Rate: {hit_rate_10:.2%} ({all_metrics[method]['hits_10']}/{total_questions})\n"
        summary_text += f"Top-10 MRR: {mrr_10:.4f}\n"
        summary_text += "-" * 25 + "\n"
        summary_text += f"--- Top {max_k} Metrics ---\n"
        summary_text += f"Top-{max_k} Hit Rate: {hit_rate_max_k:.2%} ({all_metrics[method]['hits_max_k']}/{total_questions})\n"
        summary_text += f"Top-{max_k} MRR: {mrr_max_k:.4f}\n"
        summary_text += "="*50 + "\n"
        
        print(summary_text)
        unified_log_content += summary_text
    
        log_file = os.path.join("results", f"evaluation_log_{method}_{timestamp}.txt")
    
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(f"Retrieval Method: {method.upper()}\n\n")
            f.write("=== Evaluation Results ===\n")
            f.write(f"Total Chunks Sampled: {len(chunks)}\n")
            f.write(f"Total Questions Evaluated: {total_questions}\n")
            f.write("-" * 25 + "\n")
            f.write("--- Top 1 Metrics ---\n")
            f.write(f"Top-1 Hit Rate: {hit_rate_1:.2%} ({all_metrics[method]['hits_1']}/{total_questions})\n")
            f.write("-" * 25 + "\n")
            f.write("--- Top 5 Metrics ---\n")
            f.write(f"Top-5 Hit Rate: {hit_rate_5:.2%} ({all_metrics[method]['hits_5']}/{total_questions})\n")
            f.write(f"Top-5 MRR: {mrr_5:.4f}\n")
            f.write("-" * 25 + "\n")
            f.write("--- Top 10 Metrics ---\n")
            f.write(f"Top-10 Hit Rate: {hit_rate_10:.2%} ({all_metrics[method]['hits_10']}/{total_questions})\n")
            f.write(f"Top-10 MRR: {mrr_10:.4f}\n")
            f.write("-" * 25 + "\n")
            f.write(f"--- Top {max_k} Metrics ---\n")
            f.write(f"Top-{max_k} Hit Rate: {hit_rate_max_k:.2%} ({all_metrics[method]['hits_max_k']}/{total_questions})\n")
            f.write(f"Top-{max_k} MRR: {mrr_max_k:.4f}\n")
            f.write("="*50 + "\n\n")
            f.write("--- Detailed Log ---\n")
            for i, log in enumerate(all_metrics[method]["results_log"], 1):
                if log['rank'] > 0:
                    status = f"✅ HIT (Rank: {log['rank']})"
                else:
                    status = f"❌ MISS (Not in top-{max_k})"
                    
                f.write(f"Q{i}: {log['question']}\n")
                f.write(f"   Source: {log['expected_source']}\n")
                f.write(f"   Result: {status}\n\n")

    # 통합 로그 작성
    with open(unified_log_file, "w", encoding="utf-8") as f:
        f.write(unified_log_content)

    # === 시각화 (Visualization) ===
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        # 폰트 깨짐 방지
        plt.rcParams['font.family'] = 'Malgun Gothic' if os.name == 'nt' else 'AppleGothic'
        plt.rcParams['axes.unicode_minus'] = False
        
        metrics_names = ["Top-1 Hit", "Top-5 Hit", "Top-10 Hit", f"MRR@{min(max_k, 5)}", f"MRR@{min(max_k, 10)}"]
        method_labels = [m.upper() for m in methods]
        
        # 데이터 수집
        hit1_data, hit5_data, hit10_data, mrr5_data, mrr10_data = [], [], [], [], []
        tq = total_questions if total_questions > 0 else 1
        for method in methods:
            hit1_data.append(all_metrics[method]["hits_1"] / tq)
            hit5_data.append(all_metrics[method]["hits_5"] / tq)
            hit10_data.append(all_metrics[method]["hits_10"] / tq)
            mrr5_data.append(all_metrics[method]["mrr_sum_5"] / tq)
            mrr10_data.append(all_metrics[method]["mrr_sum_10"] / tq)
            
        x = np.arange(len(metrics_names))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # 각 method별로 막대 그리기
        ax.bar(x - width, [hit1_data[0], hit5_data[0], hit10_data[0], mrr5_data[0], mrr10_data[0]], width, label=method_labels[0], color='#1f77b4')
        ax.bar(x,        [hit1_data[1], hit5_data[1], hit10_data[1], mrr5_data[1], mrr10_data[1]], width, label=method_labels[1], color='#ff7f0e')
        if len(methods) > 2:
            ax.bar(x + width, [hit1_data[2], hit5_data[2], hit10_data[2], mrr5_data[2], mrr10_data[2]], width, label=method_labels[2], color='#2ca02c')
            
        ax.set_ylabel('Scores')
        ax.set_title('RAG 정량 평가 지표 비교')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_names)
        ax.set_ylim(0, 1.1)
        ax.legend()
        
        # 텍스트 라벨 추가
        for p in ax.patches:
            height = p.get_height()
            if height > 0:
                ax.annotate(f'{height:.2f}', (p.get_x() + p.get_width() / 2., height), ha='center', va='bottom', xytext=(0, 3), textcoords='offset points', fontsize=9)
            
        fig.tight_layout()
        chart_file = os.path.join("results", f"eval_metrics_chart_{timestamp}.png")
        plt.savefig(chart_file, dpi=300)
        print(f"\n시각화 차트가 저장되었습니다: {chart_file}")
        
    except ImportError:
        print("\n[알림] matplotlib이 설치되어 있지 않아 시각화 차트를 생성하지 못했습니다. 'pip install matplotlib'을 실행해주세요.")
    except Exception as e:
        print(f"\n[오류] 차트 생성 중 오류 발생: {e}")

if __name__ == "__main__":
    # You can adjust the number of samples and max_k value here
    run_evaluation(num_samples=50, max_k=20)
