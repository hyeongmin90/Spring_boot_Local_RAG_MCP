import os
import sys
import numpy as np
from itertools import combinations
from langchain_openai import OpenAIEmbeddings

# 부모 디렉토리를 경로에 추가하여 모듈을 임포트할 수 있게 함
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

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
    # 문서들 간의 모든 쌍(pair)에 대해 코사인 유사도 계산
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

