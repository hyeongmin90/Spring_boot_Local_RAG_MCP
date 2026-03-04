import re
import threading
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank

from final_pipeline.storage import get_vectorstore, _vectorstore_lock

_bm25_retrievers = {}

def get_hybrid_reranker_retriever(k: int = 5, category: str = None, collection_name: str = "spring_docs"):
    """
    Returns the ultimate retrieval pipeline:
    Hybrid Search (Chroma Dense 0.7 + BM25 Sparse 0.3) combining into fetch_k candidates,
    followed by Cohere Reranker selecting the final Top K.
    """
    global _bm25_retrievers
    vectorstore = get_vectorstore(collection_name)
    
    # Reranker needs a larger pool from the base ensemble search
    fetch_k = max(k * 2, 30)

    # 1. Setup Chroma (Dense) Retriever
    search_kwargs = {"k": fetch_k}
    if category:
        search_kwargs["filter"] = {"category": category}
    chroma_retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)
    
    # 2. Setup BM25 (Sparse) Retriever 
    if collection_name not in _bm25_retrievers:
        with _vectorstore_lock:
            if collection_name not in _bm25_retrievers:
                db_data = vectorstore.get()
                if not db_data or not db_data.get('documents'):
                    return chroma_retriever # Fallback if empty
                
                docs = []
                for idx in range(len(db_data['documents'])):
                    content = db_data['documents'][idx]
                    metadata = db_data['metadatas'][idx] if 'metadatas' in db_data else {}
                    docs.append(Document(page_content=content, metadata=metadata))
                
                def preprocess_text(text: str) -> list[str]:
                    words = re.findall(r'\b\w+\b', text.lower())
                    stopwords = {"the", "a", "an", "is", "in", "it", "to", "of", "and", "or", "for", "with", "on", "by", "this", "that"}
                    return [w for w in words if w not in stopwords]
                
                _bm25_retrievers[collection_name] = BM25Retriever.from_documents(
                    docs, preprocess_func=preprocess_text
                )
    
    _bm25_retrievers[collection_name].k = fetch_k
    
    # 3. Combine into Ensemble Retriever (RRF 0.7 / 0.3)
    ensemble_retriever = EnsembleRetriever(
        retrievers=[chroma_retriever, _bm25_retrievers[collection_name]],
        weights=[0.7, 0.3]
    )
    
    # 4. Attach Cohere Reranker Pipeline
    compressor = CohereRerank(model="rerank-english-v3.0", top_n=k)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=ensemble_retriever
    )
    
    return compression_retriever

def query(query_text: str, k: int = 5, category: str = None) -> list[Document]:
    """Execute the final hybrid+reranker retrieval pipeline."""
    retriever = get_hybrid_reranker_retriever(k=k, category=category)
    if hasattr(retriever, 'invoke'):
        return retriever.invoke(query_text)
    return retriever.get_relevant_documents(query_text)
