# import chromadb
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
# Initialize ChromaDB client
import threading
import re

# Global vectorstore instance (Singleton)
_vectorstore_lock = threading.Lock()
_vectorstores = {}
_bm25_retrievers = {}

PERSIST_DIRECTORY = "./chroma_db"

from tqdm import tqdm

def get_vectorstore(collection_name="spring_docs"):
    """
    Returns the initialized Chroma vectorstore instance (Singleton).
    Thread-safe initialization.
    """
    global _vectorstores
    
    if collection_name not in _vectorstores:
        with _vectorstore_lock:
            if collection_name not in _vectorstores:
                # Initialize
                embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
                _vectorstores[collection_name] = Chroma(
                    collection_name=collection_name,
                    embedding_function=embeddings,
                    persist_directory=PERSIST_DIRECTORY,
                )
                
    return _vectorstores[collection_name]

def add_documents(documents, collection_name="spring_docs"):
    """
    Adds a list of Document objects to the vectorstore.
    """
    if not documents:
        tqdm.write("No documents to add.")
        return

    vectorstore = get_vectorstore(collection_name)
    tqdm.write(f"Adding {len(documents)} documents to ChromaDB ({collection_name})...")
    url_link = documents[0].metadata["source"]
    result = vectorstore.get(where={"source": url_link})
    ids_to_delete = result["ids"]

    if ids_to_delete:
        vectorstore.delete(ids=ids_to_delete)

    ids = [doc.metadata["chunk_id"] for doc in documents]
    vectorstore.add_documents(documents=documents, ids=ids)
    tqdm.write("Documents added successfully.")

def mmr_query_documents(query, k=3, category=None, collection_name="spring_docs", lambda_mult=0.5, fetch_k=20):
    """
    Searches for documents similar to the query. with mmr
    """
    vectorstore = get_vectorstore(collection_name)
    search_filter = None
    if category:
        search_filter = {"category": category}
    
    results = vectorstore.max_marginal_relevance_search(
        query=query, 
        k=k, 
        filter=search_filter,
        lambda_mult=lambda_mult,
        fetch_k=fetch_k
    )
    
    return results


def query_documents(query, k=3, category=None, collection_name="spring_docs"):
    """
    Searches for documents similar to the query.
    """
    vectorstore = get_vectorstore(collection_name)
    search_filter = None
    if category:
        search_filter = {"category": category}
    
    results = vectorstore.similarity_search(query, k=k, filter=search_filter)
    return results

# Add support for Hybrid Search combining Chroma (Dense) and BM25 (Sparse)
# import warnings
# with warnings.catch_warnings():
#     warnings.simplefilter("ignore")
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank


_bm25_retriever = None

def get_hybrid_retriever(k=3, category=None, collection_name="spring_docs", dense_weight=0.7, sparse_weight=0.3, use_reranker=False):
    """
    Returns an EnsembleRetriever combining dense (Chroma) and sparse (BM25) retrievers.
    """
    global _bm25_retrievers
    
    vectorstore = get_vectorstore(collection_name)
    
    # Reranker needs more candidates from base retrievers (dense and bm25)
    fetch_k = max(k * 2, 30) if use_reranker else k

    # Setup Chroma Retriever
    search_kwargs = {"k": fetch_k}
    if category:
        search_kwargs["filter"] = {"category": category}
    
    chroma_retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)
    
    # Initialize BM25 Retriever by fetching all documents from Chroma
    # We do this lazily (only when hybrid search is requested)
    if collection_name not in _bm25_retrievers:
        with _vectorstore_lock:
            if collection_name not in _bm25_retrievers:
                tqdm.write(f"Initializing BM25 Retriever from Chroma documents ({collection_name})...")
                db_data = vectorstore.get()
                
                if not db_data or not db_data.get('documents'):
                    tqdm.write("No documents found in Chroma to build BM25 retriever.")
                    return chroma_retriever # Fallback to just Chroma if empty
                
                # Reconstruct Document objects
                docs = []
                for idx in range(len(db_data['documents'])):
                    content = db_data['documents'][idx]
                    metadata = db_data['metadatas'][idx] if 'metadatas' in db_data else {}
                    docs.append(Document(page_content=content, metadata=metadata))
                
                # Define a preprocessing function for BM25
                def preprocess_text(text: str) -> list[str]:
                    text = text.lower()
                    # Keep words and numbers
                    words = re.findall(r'\b\w+\b', text)
                    stopwords = {
                        "the", "a", "an", "is", "in", "it", "to", "of", "and", "or",
                        "for", "with", "on", "by", "this", "that", "these", "those",
                        "we", "you", "they", "he", "she", "at", "from", "as", "be",
                        "are", "was", "were", "has", "have", "had", "do", "does", "did",
                        "but", "not", "can", "could", "would", "should", "what", "how",
                        "where", "when", "why", "who", "which"
                    }
                    return [w for w in words if w not in stopwords]
                
                # Create BM25 retriever with custom preprocessing
                _bm25_retrievers[collection_name] = BM25Retriever.from_documents(
                    docs,
                    preprocess_func=preprocess_text
                )
    
    # Set fetch size for BM25
    _bm25_retrievers[collection_name].k = fetch_k
    
    # Create the Ensemble Retriever
    ensemble_retriever = EnsembleRetriever(
        retrievers=[chroma_retriever, _bm25_retrievers[collection_name]],
        weights=[dense_weight, sparse_weight]
    )
    
    if use_reranker:
        # Initialize Cohere Reranker
        compressor = CohereRerank(model="rerank-english-v3.0", top_n=k)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=ensemble_retriever
        )
        return compression_retriever
        
    return ensemble_retriever

def query_hybrid(query, k=3, category=None, collection_name="spring_docs", dense_weight=0.7, sparse_weight=0.3, use_reranker=False):
    """
    Searches for documents using Hybrid Search (BM25 + Chroma Dense).
    """
    retriever = get_hybrid_retriever(k=k, category=category, collection_name=collection_name, dense_weight=dense_weight, sparse_weight=sparse_weight, use_reranker=use_reranker)
    if hasattr(retriever, 'invoke'):
         results = retriever.invoke(query)
    else:
         results = retriever.get_relevant_documents(query)
    return results[:k]

if __name__ == "__main__":
    from dotenv import load_dotenv
    import sys
    import os
    
    # Add project root to sys.path
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    load_dotenv()
    
    print("=== Vector Store Test Console ===")
    print("Type 'exit' or 'q' to quit.")
    
    while True:
        query = input("\nEnter query: ").strip()
        if query.lower() in ('exit', 'q'):
            break
            
        if not query:
            continue
            
        k_str = input("How many results (k) [default 3]: ").strip()
        k = int(k_str) if k_str.isdigit() else 3
        docs_type_str = input("Filter by docs_type (spring-boot or spring-data-redis) [default None]: ").strip()
        docs_type = None
        if docs_type_str:
            docs_type = docs_type_str
        try:
            results = query_documents(query, k=k, docs_type=docs_type)
            print(f"\nFound {len(results)} results:")
            for i, doc in enumerate(results):
                source = doc.metadata.get("source", "Unknown")
                original_content = doc.metadata.get("original_content", "")
                print(f"\n[{i+1}] Source: {source}")
                print(f"     Content: {doc.page_content}")
                print(f"     Original Content: {original_content}")
        except Exception as e:
            print(f"Error querying: {e}")
