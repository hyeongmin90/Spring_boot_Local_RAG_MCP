import chromadb
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
# Initialize ChromaDB client
import threading

# Global vectorstore instance (Singleton)
_vectorstore_lock = threading.Lock()
_vectorstore = None

PERSIST_DIRECTORY = "./chroma_db"

from tqdm import tqdm

def get_vectorstore():
    """
    Returns the initialized Chroma vectorstore instance (Singleton).
    Thread-safe initialization.
    """
    global _vectorstore
    
    if _vectorstore is None:
        with _vectorstore_lock:
            if _vectorstore is None:
                # Initialize
                embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
                _vectorstore = Chroma(
                    collection_name="spring_docs",
                    embedding_function=embeddings,
                    persist_directory=PERSIST_DIRECTORY,
                )
                
    return _vectorstore

def add_documents(documents):
    """
    Adds a list of Document objects to the vectorstore.
    """
    if not documents:
        tqdm.write("No documents to add.")
        return

    vectorstore = get_vectorstore()
    tqdm.write(f"Adding {len(documents)} documents to ChromaDB...")
    url_link = documents[0].metadata["source"]
    result = vectorstore.get(where={"source": url_link})
    ids_to_delete = result["ids"]

    if ids_to_delete:
        vectorstore.delete(ids=ids_to_delete)

    ids = [doc.metadata["chunk_id"] for doc in documents]
    vectorstore.add_documents(documents=documents, ids=ids)
    tqdm.write("Documents added successfully.")

def query_documents(query, k=3, category=None):
    """
    Searches for documents similar to the query.
    """
    vectorstore = get_vectorstore()
    search_filter = None
    if category:
        search_filter = {"category": category}
    
    results = vectorstore.similarity_search(query, k=k, filter=search_filter)
    return results

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
