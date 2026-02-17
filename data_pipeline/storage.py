import chromadb
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

# Initialize ChromaDB client
PERSIST_DIRECTORY = "./chroma_db"

def get_vectorstore():
    """
    Returns the initialized Chroma vectorstore instance.
    """
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    vectorstore = Chroma(
        collection_name="spring_boot_docs",
        embedding_function=embeddings,
        persist_directory=PERSIST_DIRECTORY,
    )
    return vectorstore

def add_documents(documents):
    """
    Adds a list of Document objects to the vectorstore.
    """
    if not documents:
        print("No documents to add.")
        return

    vectorstore = get_vectorstore()
    print(f"Adding {len(documents)} documents to ChromaDB...")
    vectorstore.add_documents(documents)
    print("Documents added successfully.")

def query_documents(query, k=3):
    """
    Searches for documents similar to the query.
    """
    vectorstore = get_vectorstore()
    results = vectorstore.similarity_search(query, k=k)
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
        
        try:
            results = query_documents(query, k=k)
            print(f"\nFound {len(results)} results:")
            for i, doc in enumerate(results):
                source = doc.metadata.get("source", "Unknown")
                print(f"\n[{i+1}] Source: {source}")
                print(f"     Content: {doc.page_content}")
        except Exception as e:
            print(f"Error querying: {e}")
