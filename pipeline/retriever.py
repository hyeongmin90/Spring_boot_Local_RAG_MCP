import threading

from tqdm import tqdm
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_cohere import CohereRerank
import re

from pipeline.storage import get_vectorstore

_vectorstore_lock = threading.Lock()
_bm25_retrievers: dict = {}


def get_hybrid_retriever(
    k: int = 3,
    category: str = None,
    collection_name: str = "spring_docs",
    dense_weight: float = 0.7,
    sparse_weight: float = 0.3,
    use_reranker: bool = False,
):
    """
    Dense(Chroma) + Sparse(BM25) EnsembleRetriever를 반환합니다.
    use_reranker=True일 경우 Cohere Reranker를 최상단에 추가합니다.
    """
    global _bm25_retrievers

    vectorstore = get_vectorstore(collection_name)
    fetch_k = max(k * 2, 30) if use_reranker else k

    search_kwargs: dict = {"k": fetch_k}
    if category:
        search_kwargs["filter"] = {"category": category}

    chroma_retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)

    bm25_key = f"{collection_name}_{category}" if category else collection_name

    if bm25_key not in _bm25_retrievers:
        with _vectorstore_lock:
            if bm25_key not in _bm25_retrievers:
                tqdm.write(f"Initializing BM25 Retriever (Key: {bm25_key})...")

                where_filter = {"category": category} if category else None
                db_data = vectorstore.get(where=where_filter)

                if not db_data or not db_data.get("documents"):
                    tqdm.write(f"No documents found for BM25 key {bm25_key}.")
                    return chroma_retriever

                docs = [
                    Document(
                        page_content=db_data["documents"][i],
                        metadata=db_data["metadatas"][i] if "metadatas" in db_data else {},
                    )
                    for i in range(len(db_data["documents"]))
                ]

                def preprocess_text(text: str) -> list[str]:
                    stopwords = {
                        "the", "a", "an", "is", "in", "it", "to", "of", "and", "or",
                        "for", "with", "on", "by", "this", "that", "these", "those",
                        "we", "you", "they", "he", "she", "at", "from", "as", "be",
                        "are", "was", "were", "has", "have", "had", "do", "does", "did",
                        "but", "not", "can", "could", "would", "should", "what", "how",
                        "where", "when", "why", "who", "which",
                    }
                    return [w for w in re.findall(r"\b\w+\b", text.lower()) if w not in stopwords]

                _bm25_retrievers[bm25_key] = BM25Retriever.from_documents(
                    docs, preprocess_func=preprocess_text
                )

    _bm25_retrievers[bm25_key].k = fetch_k

    ensemble = EnsembleRetriever(
        retrievers=[chroma_retriever, _bm25_retrievers[bm25_key]],
        weights=[dense_weight, sparse_weight],
    )

    if use_reranker:
        compressor = CohereRerank(model="rerank-english-v3.0", top_n=k)
        return ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=ensemble,
        )

    return ensemble


def query_hybrid(
    query: str,
    k: int = 3,
    category: str = None,
    collection_name: str = "spring_docs",
    dense_weight: float = 0.7,
    sparse_weight: float = 0.3,
    use_reranker: bool = False,
) -> list[Document]:
    """Hybrid Search(BM25 + Dense) 실행 후 상위 k개를 반환합니다."""
    retriever = get_hybrid_retriever(
        k=k,
        category=category,
        collection_name=collection_name,
        dense_weight=dense_weight,
        sparse_weight=sparse_weight,
        use_reranker=use_reranker,
    )
    results = retriever.invoke(query) if hasattr(retriever, "invoke") else retriever.get_relevant_documents(query)
    return results[:k]

def query_documents(query: str, k: int = 3, category: str = None, collection_name: str = "spring_docs") -> list:
    """Dense similarity search로 상위 k개 문서를 반환합니다."""
    vectorstore = get_vectorstore(collection_name)
    search_filter = {"category": category} if category else None
    return vectorstore.similarity_search(query, k=k, filter=search_filter)

def mmr_query_documents(
    query: str,
    k: int = 3,
    category: str = None,
    collection_name: str = "spring_docs",
    lambda_mult: float = 0.5,
    fetch_k: int = 20,
) -> list:
    """MMR(Maximal Marginal Relevance) 검색으로 다양성이 보장된 상위 k개 문서를 반환합니다."""
    vectorstore = get_vectorstore(collection_name)
    search_filter = {"category": category} if category else None
    return vectorstore.max_marginal_relevance_search(
        query=query,
        k=k,
        filter=search_filter,
        lambda_mult=lambda_mult,
        fetch_k=fetch_k,
    )