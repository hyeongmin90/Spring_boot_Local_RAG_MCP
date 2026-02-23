import os
import sys
import json
from tqdm import tqdm
from datetime import datetime
from langchain_openai import OpenAIEmbeddings

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_pipeline.storage import query_documents, mmr_query_documents, query_hybrid
from data_pipeline.evaluate_redundancy import calculate_semantic_redundancy, calculate_lexical_redundancy

def evaluate_retrieval(question, expected_id, method="dense", k=10):
    """
    Queries the vector store with the generated question and checks if the expected chunk is retrieved.
    Calculates the rank of the expected chunk.
    """
    if method == "mmr":
        results = mmr_query_documents(question, k=k)
    elif method == "hybrid":
        results = query_hybrid(question, k=k)
    else:
        results = query_documents(question, k=k)

    for rank, doc in enumerate(results, start=1):
        retrieved_id = getattr(doc, 'id', None) or doc.metadata.get("chunk_id")
  
        # Check if it matches the expected chunk
        # Matching by id is preferred if available, otherwise by source as a loose fallback
        if expected_id and retrieved_id == expected_id:
            return rank, results
       
            
    return -1, [] # Not found within top k

def run_comprehensive_evaluation(dataset_file="evaluation_dataset.json", max_k=50):
    print("=== RAG 종합 평가 시작 ===")
    
    if not os.path.exists(dataset_file):
        print(f"오류: 데이터셋 파일 '{dataset_file}'을 찾을 수 없습니다.")
        return
        
    with open(dataset_file, "r", encoding="utf-8") as f:
        dataset = json.load(f)
        
    print(f"데이터셋 로드 완료: {len(dataset)}개의 청크 항목")
    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small") 
    
    methods = ["dense", "hybrid"]
    
    all_metrics = {m: {
        "hits_5": 0, "mrr_sum_5": 0.0, 
        "hits_10": 0, "mrr_sum_10": 0.0, 
        "hits_max_k": 0, "mrr_sum_max_k": 0.0, 
        "total_semantic_redundancy": 0.0,
        "total_lexical_redundancy": 0.0,
        "valid_redundancy_queries": 0,
        "results_log": []
    } for m in methods}
    
    total_questions = 0
    
    for item in tqdm(dataset, desc="Evaluating Dataset"):
        expected_id = item.get("id")
        expected_source = item.get("source", "Unknown")
        questions = item.get("questions", [])
        
        for q_idx, question in enumerate(questions):
            total_questions += 1
            
            for method in methods:
                # 검색 결과 가져오기
                rank, retrieved_docs = evaluate_retrieval(question, expected_id, method=method, k=max_k)
                
                # Retrieval Metrics 업데이트
                is_hit_5 = 0 < rank <= 5
                if is_hit_5:
                    all_metrics[method]["hits_5"] += 1
                    all_metrics[method]["mrr_sum_5"] += 1.0 / rank

                is_hit_10 = 0 < rank <= 10
                if is_hit_10:
                    all_metrics[method]["hits_10"] += 1
                    all_metrics[method]["mrr_sum_10"] += 1.0 / rank

                is_hit_max_k = 0 < rank <= max_k
                if is_hit_max_k:
                    all_metrics[method]["hits_max_k"] += 1
                    all_metrics[method]["mrr_sum_max_k"] += 1.0 / rank
                    
                # 중복도 평가 (Top-k 기준, 반환된 문서들이 2개 이상일 때)
                if len(retrieved_docs) > 1:
                    sem_red = calculate_semantic_redundancy(retrieved_docs, embeddings_model)
                    lex_red = calculate_lexical_redundancy(retrieved_docs)
                    
                    all_metrics[method]["total_semantic_redundancy"] += sem_red
                    all_metrics[method]["total_lexical_redundancy"] += lex_red
                    all_metrics[method]["valid_redundancy_queries"] += 1

                all_metrics[method]["results_log"].append({
                    "chunk_id": expected_id,
                    "question": question,
                    "expected_source": expected_source,
                    "rank": rank,
                })
                
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 평가 결과 출력 및 저장
    for method in methods:
        tq = total_questions if total_questions > 0 else 1
        vq = all_metrics[method]["valid_redundancy_queries"] if all_metrics[method]["valid_redundancy_queries"] > 0 else 1
        
        hit_rate_5 = all_metrics[method]["hits_5"] / tq
        mrr_5 = all_metrics[method]["mrr_sum_5"] / tq
        hit_rate_10 = all_metrics[method]["hits_10"] / tq
        mrr_10 = all_metrics[method]["mrr_sum_10"] / tq
        hit_rate_max_k = all_metrics[method]["hits_max_k"] / tq
        mrr_max_k = all_metrics[method]["mrr_sum_max_k"] / tq
        
        avg_sem_red = all_metrics[method]["total_semantic_redundancy"] / vq
        avg_lex_red = all_metrics[method]["total_lexical_redundancy"] / vq
        
        print("\n" + "="*50)
        print(f"=== 평가 결과 ({method.upper()}) ===")
        print(f"평가된 총 질문 수: {total_questions}")
        print("-" * 25)
        print("--- 검색 성능 (Retrieval Metrics) ---")
        print(f"Top-5 Hit Rate: {hit_rate_5:.2%} | MRR: {mrr_5:.4f}")
        print(f"Top-10 Hit Rate: {hit_rate_10:.2%} | MRR: {mrr_10:.4f}")
        print(f"Top-{max_k} Hit Rate: {hit_rate_max_k:.2%} | MRR: {mrr_max_k:.4f}")
        print("-" * 25)
        print("--- 검색 결과 중복도 (Redundancy Metrics) ---")
        print(f"의미적 중복도(Semantic Redundancy): {avg_sem_red:.4f}")
        print(f"어휘적 중복도(Lexical Redundancy): {avg_lex_red:.4f}")
        print("="*50)

        log_file = f"comprehensive_eval_log_{method}_{timestamp}.txt"
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(f"Retrieval Method: {method.upper()}\n\n")
            f.write("=== Evaluation Results ===\n")
            f.write(f"Total Questions Evaluated: {total_questions}\n")
            f.write("-" * 25 + "\n")
            f.write("--- Retrieval Metrics ---\n")
            f.write(f"Top-5 Hit Rate: {hit_rate_5:.2%} ({all_metrics[method]['hits_5']}/{total_questions}) | MRR: {mrr_5:.4f}\n")
            f.write(f"Top-10 Hit Rate: {hit_rate_10:.2%} ({all_metrics[method]['hits_10']}/{total_questions}) | MRR: {mrr_10:.4f}\n")
            f.write(f"Top-{max_k} Hit Rate: {hit_rate_max_k:.2%} ({all_metrics[method]['hits_max_k']}/{total_questions}) | MRR: {mrr_max_k:.4f}\n")
            f.write("-" * 25 + "\n")
            f.write("--- Redundancy Metrics ---\n")
            f.write(f"Semantic Redundancy: {avg_sem_red:.4f}\n")
            f.write(f"Lexical Redundancy: {avg_lex_red:.4f}\n")
            f.write("="*50 + "\n\n")
            f.write("--- Detailed Log ---\n")
            for i, log in enumerate(all_metrics[method]["results_log"], 1):
                if log['rank'] > 0 and log['rank'] <= max_k:
                    status = f"✅ HIT (Rank: {log['rank']})"
                else:
                    status = f"❌ MISS (Not in top-{max_k})"
                f.write(f"Q{i}: {log['question']}\n")
                f.write(f"   Source: {log['expected_source']}\n")
                f.write(f"   Result: {status}\n\n")

if __name__ == "__main__":
    run_comprehensive_evaluation(dataset_file="evaluation_dataset.json", max_k=50)
