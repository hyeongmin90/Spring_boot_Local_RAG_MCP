import os
import sys
import json
from tqdm import tqdm
from datetime import datetime
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

load_dotenv()

from langchain_openai import OpenAIEmbeddings
from data_pipeline.storage import query_documents, mmr_query_documents, query_hybrid
from data_pipeline.evaluation.evaluate_redundancy import calculate_semantic_redundancy, calculate_lexical_redundancy

def evaluate_retrieval(question, expected_id, method="dense", k=10):
    """
    Queries the vector store with the generated question and checks if the expected chunk is retrieved.
    Calculates the rank of the expected chunk.
    """
    if method.startswith("mmr"):
        # Format: "mmr" or "mmr_{lambda_mult}_{fetch_k}"
        parts = method.split("_")
        lm = 0.5
        fk = max(20, k * 2)
        if len(parts) == 3:
            lm = float(parts[1])
            fk = int(parts[2])
        results = mmr_query_documents(question, k=k, lambda_mult=lm, fetch_k=fk)
    elif method.startswith("hybrid_cohere"):
        # Format: "hybrid_cohere" or "hybrid_cohere_{dense_weight}_{sparse_weight}"
        parts = method.split("_")
        dw = 0.7
        sw = 0.3
        if len(parts) == 4:
            dw = float(parts[2])
            sw = float(parts[3])
        results = query_hybrid(question, k=k, dense_weight=dw, sparse_weight=sw, use_reranker=True)
    elif method.startswith("hybrid"):
        # Format: "hybrid" or "hybrid_{dense_weight}_{sparse_weight}"
        parts = method.split("_")
        dw = 0.7
        sw = 0.3
        if len(parts) == 3:
            dw = float(parts[1])
            sw = float(parts[2])
        results = query_hybrid(question, k=k, dense_weight=dw, sparse_weight=sw, use_reranker=False)
    else:
        results = query_documents(question, k=k)

    for rank, doc in enumerate(results, start=1):
        retrieved_id = getattr(doc, 'id', None) or doc.metadata.get("chunk_id")
  
        # Check if it matches the expected chunk
        # Matching by id is preferred if available, otherwise by source as a loose fallback
        if expected_id and retrieved_id == expected_id:
            return rank, results
            
    return -1, [] # Not found within top k

def run_comprehensive_evaluation(dataset_file="evaluation_dataset_split_7.json", max_k=50):
    print("=== RAG 종합 평가 시작 ===")
    
    if not os.path.exists(dataset_file):
        print(f"오류: 데이터셋 파일 '{dataset_file}'을 찾을 수 없습니다.")
        return
        
    with open(dataset_file, "r", encoding="utf-8") as f:
        dataset = json.load(f)
        
    print(f"데이터셋 로드 완료: {len(dataset)}개의 청크 항목")
    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small") 
    
    #============================
    # Test different Hybrid ratios 
    # Format: hybrid_{dense_weight}_{sparse_weight} 
    methods = [
        "dense",          # Baseline Cosine
        "hybrid",         # Default (Dense 0.7, Sparse 0.3)
        "hybrid_0.5_0.5", # Equal Weight
        "hybrid_0.3_0.7", # Sparse Heavy
        "hybrid_cohere",  # Cohere Reranker
    ]
    #============================

    all_metrics = {m: {
        "hits_1": 0,
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
                is_hit_1 = rank == 1
                if is_hit_1:
                    all_metrics[method]["hits_1"] += 1

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
    os.makedirs("results", exist_ok=True)
    unified_log_file = os.path.join("results", f"comprehensive_eval_summary_{timestamp}.txt")
    unified_log_content = f"=== RAG 중합 평가 요약 ({timestamp}) ===\n"
    unified_log_content += f"평가된 총 질문 수: {total_questions}\n"
    
    # 평가 결과 출력 및 저장
    for method in methods:
        tq = total_questions if total_questions > 0 else 1
        vq = all_metrics[method]["valid_redundancy_queries"] if all_metrics[method]["valid_redundancy_queries"] > 0 else 1
        
        hit_rate_1 = all_metrics[method]["hits_1"] / tq
        hit_rate_5 = all_metrics[method]["hits_5"] / tq
        mrr_5 = all_metrics[method]["mrr_sum_5"] / tq
        hit_rate_10 = all_metrics[method]["hits_10"] / tq
        mrr_10 = all_metrics[method]["mrr_sum_10"] / tq
        hit_rate_max_k = all_metrics[method]["hits_max_k"] / tq
        mrr_max_k = all_metrics[method]["mrr_sum_max_k"] / tq
        
        avg_sem_red = all_metrics[method]["total_semantic_redundancy"] / vq
        avg_lex_red = all_metrics[method]["total_lexical_redundancy"] / vq
        
        # 통합 로그 및 콘솔용 텍스트
        summary_text = f"\n" + "="*50 + f"\n=== 평가 결과 ({method.upper()}) ===\n"
        summary_text += "-" * 25 + "\n"
        summary_text += "--- 검색 성능 (Retrieval Metrics) ---\n"
        summary_text += f"Top-1 Hit Rate: {hit_rate_1:.2%}\n"
        summary_text += f"Top-5 Hit Rate: {hit_rate_5:.2%} | MRR: {mrr_5:.4f}\n"
        summary_text += f"Top-10 Hit Rate: {hit_rate_10:.2%} | MRR: {mrr_10:.4f}\n"
        summary_text += f"Top-{max_k} Hit Rate: {hit_rate_max_k:.2%} | MRR: {mrr_max_k:.4f}\n"
        summary_text += "-" * 25 + "\n"
        summary_text += "--- 검색 결과 중복도 (Redundancy Metrics) ---\n"
        summary_text += f"의미적 중복도(Semantic Redundancy): {avg_sem_red:.4f}\n"
        summary_text += f"어휘적 중복도(Lexical Redundancy): {avg_lex_red:.4f}\n"
        summary_text += "="*50 + "\n"
        
        print(summary_text)
        unified_log_content += summary_text

        # 개별 상세 로그 (results 디렉토리에 저장)
        log_file = os.path.join("results", f"comprehensive_eval_log_{method}_{timestamp}.txt")
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(f"Retrieval Method: {method.upper()}\n\n")
            f.write("=== Evaluation Results ===\n")
            f.write(f"Total Questions Evaluated: {total_questions}\n")
            f.write("-" * 25 + "\n")
            f.write("--- Retrieval Metrics ---\n")
            f.write(f"Top-1 Hit Rate: {hit_rate_1:.2%} ({all_metrics[method]['hits_1']}/{total_questions})\n")
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
        
        fig, ax = plt.subplots(figsize=(14, 7))
        
        # 동적으로 N개의 method를 위한 그래프 막대 폭과 오프셋 계산
        num_methods = len(methods)
        width = 0.8 / num_methods
        offsets = np.linspace(-width*(num_methods-1)/2, width*(num_methods-1)/2, num_methods)
        
        # 각 method별로 막대 그리기
        import matplotlib.cm as cm
        colors = cm.get_cmap('tab10').colors
        
        for i, method in enumerate(methods):
            data = [hit1_data[i], hit5_data[i], hit10_data[i], mrr5_data[i], mrr10_data[i]]
            ax.bar(x + offsets[i], data, width, label=method_labels[i], color=colors[i % len(colors)])
            
        ax.set_ylabel('Scores')
        ax.set_title('RAG 종합 평가 지표 비교')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_names)
        ax.set_ylim(0, 1.1) # Y축은 0~1.1 (최대값 1이므로)
        ax.legend()
        
        # 텍스트 라벨 추가
        for p in ax.patches:
            height = p.get_height()
            if height > 0:
                ax.annotate(f'{height:.2f}', (p.get_x() + p.get_width() / 2., height), ha='center', va='bottom', xytext=(0, 3), textcoords='offset points', fontsize=9)
            
        fig.tight_layout()
        chart_file = os.path.join("results", f"comprehensive_eval_chart_{timestamp}.png")
        plt.savefig(chart_file, dpi=300)
        print(f"\n시각화 차트가 저장되었습니다: {chart_file}")
        
    except ImportError:
        print("\n[알림] matplotlib이 설치되어 있지 않아 시각화 차트를 생성하지 못했습니다. 'pip install matplotlib'을 실행해주세요.")
    except Exception as e:
        print(f"\n[오류] 차트 생성 중 오류 발생: {e}")

if __name__ == "__main__":
    run_comprehensive_evaluation(dataset_file="evaluation_dataset_split_3.json", max_k=20)
