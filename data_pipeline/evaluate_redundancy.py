import os
import sys
import numpy as np
import concurrent.futures
from tqdm import tqdm
from itertools import combinations
from datetime import datetime
from langchain_openai import OpenAIEmbeddings

# 부모 디렉토리를 경로에 추가하여 모듈을 임포트할 수 있게 함
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_pipeline.storage import query_documents
from data_pipeline.evaluate_rag import get_random_chunks, generate_questions

def cosine_similarity(vec_a, vec_b):
    """
    두 벡터 간의 코사인 유사도를 계산합니다.
    """
    dot_product = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)

def calculate_semantic_redundancy(documents, embeddings_model):
    """
    임베딩 모델을 사용하여 문서들 간의 평균 쌍별 코사인 유사도를 계산해 의미적 중복도를 측정합니다.
    """
    if len(documents) < 2:
        return 0.0
        
    texts = [doc.page_content for doc in documents]
    # 모든 추출된 문서의 임베딩을 가져옴
    embeddings = embeddings_model.embed_documents(texts)
    
    similarities = []
    # 문서들 간의 모든 쌍(pair)에 대해 Ко사인 유사도 계산
    for (i, j) in combinations(range(len(embeddings)), 2):
        sim = cosine_similarity(embeddings[i], embeddings[j])
        similarities.append(sim)
        
    return np.mean(similarities) if similarities else 0.0

def calculate_lexical_redundancy(documents):
    """
    단순 어휘적 중복도를 자카드 유사도(Jaccard Similarity)를 사용해 계산합니다.
    """
    if len(documents) < 2:
        return 0.0
        
    tokenized_docs = [set(doc.page_content.lower().split()) for doc in documents]
    
    similarities = []
    # 문서들 간의 모든 쌍(pair)에 대해 자카드 유사도 계산
    for (i, j) in combinations(range(len(tokenized_docs)), 2):
        set1 = tokenized_docs[i]
        set2 = tokenized_docs[j]
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        sim = intersection / union if union > 0 else 0
        similarities.append(sim)
        
    return np.mean(similarities) if similarities else 0.0

def evaluate_retrieval_redundancy(num_samples=10, k=5):
    """
    무작위 청크를 이용해 질문을 생성하고, 생성된 질문으로 검색된 문서의 중복도를 평가합니다.
    """
    print("=== 중복률(Redundancy) 평가 시작 ===")
    print(f"{num_samples}개의 청크 샘플링 및 질문 생성 중...")
    chunks = get_random_chunks(n=num_samples)
    if not chunks:
        print("청크를 찾지 못했습니다.")
        return
        
    # 의미적 중복도를 측정하기 위한 임베딩 모델
    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small") 
    
    questions_list = []
    
    # 1. 평가용 질문 생성
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_chunk = {executor.submit(generate_questions, item['content']): item for item in chunks}
        for future in tqdm(concurrent.futures.as_completed(future_to_chunk), total=len(chunks), desc="LLM Question Generation"):
            try:
                questions = future.result()
                questions_list.extend(questions)
            except Exception as e:
                print(f"질문 생성 중 오류 발생: {e}")
                
    total_semantic_redundancy = 0.0
    total_lexical_redundancy = 0.0
    valid_queries = 0
    
    total_questions = len(questions_list)
    print(f"\n생성된 {total_questions}개의 질문으로 Top-{k} Dense 검색의 중복률 평가 중...")
    
    # 2. 검색 수행 및 중복도 계산
    for q in tqdm(questions_list, desc="Measuring Redundancy"):
        # Dense 검색 (Vector similarity search)
        retrieved_docs = query_documents(q, k=k)
        
        # 2개 이상 검색되었을 때만 중복도 계산 가능
        if len(retrieved_docs) > 1:
            sem_red = calculate_semantic_redundancy(retrieved_docs, embeddings_model)
            lex_red = calculate_lexical_redundancy(retrieved_docs)
            
            total_semantic_redundancy += sem_red
            total_lexical_redundancy += lex_red
            valid_queries += 1
            
    if valid_queries == 0:
        print("\n유효한 검색 결과를 얻지 못해 평가를 진행할 수 없습니다.")
        return
        
    avg_sem_red = total_semantic_redundancy / valid_queries
    avg_lex_red = total_lexical_redundancy / valid_queries
    
    with open("redundancy_results.txt", "w", encoding="utf-8") as f:
        f.write("=== 중복률 평가 결과 ===\n")
        f.write("Retrieval Method: Dense Retrieval (Similarity Search)\n")
        f.write(f"평가된 질문 수: {valid_queries}\n")
        f.write(f"Top-K 설정: {k}\n")
        f.write("-" * 25 + "\n")
        f.write(f"의미적 중복도(Semantic Redundancy): {avg_sem_red:.4f}\n")
        f.write("  * 문서 간 쌍별 코사인 유사도 평균입니다.\n")
        f.write("  * 높은 값 (1.0에 가까움) -> 검색된 문서들의 의미가 매우 비슷함.\n")
        f.write("  * 낮은 값 -> 검색된 문서가 다양한 문맥/내용을 담고 있음.\n")
        f.write("-" * 25 + "\n")
        f.write(f"어휘적 중복도(Lexical Redundancy): {avg_lex_red:.4f}\n")
        f.write("  * 문서 간 쌍별 자카드 유사도(Jaccard) 평균입니다.\n")
        f.write("  * 높은 값 -> 정확히 일치하는 단어를 많이 공유하고 있음.\n")
        f.write("="*50 + "\n")
    
    print("saved result: redundancy_results.txt")

if __name__ == "__main__":
    # 샘플 개수와 Top-K 설정을 여기서 변경하실 수 있습니다.
    evaluate_retrieval_redundancy(num_samples=100, k=10)
