import threading
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

_vectorstore_lock = threading.Lock()
_vectorstores = {}
PERSIST_DIRECTORY = "./chroma_db"

def get_vectorstore(collection_name: str = "spring_docs") -> Chroma:
    """
    Returns the initialized Chroma vectorstore instance (Singleton).
    """
    global _vectorstores
    
    if collection_name not in _vectorstores:
        with _vectorstore_lock:
            if collection_name not in _vectorstores:
                embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
                _vectorstores[collection_name] = Chroma(
                    collection_name=collection_name,
                    embedding_function=embeddings,
                    persist_directory=PERSIST_DIRECTORY,
                )
    return _vectorstores[collection_name]

def add_documents(documents: list, collection_name: str = "spring_docs"):
    """
    Adds a list of Document objects to the vectorstore.
    Deletes existing documents with the same source URL before adding to prevent duplication.
    """
    if not documents:
        return

    vectorstore = get_vectorstore(collection_name)
    url_link = documents[0].metadata["source"]
    
    # Remove older chunks from the same source to avoid duplication
    result = vectorstore.get(where={"source": url_link})
    ids_to_delete = result["ids"]
    if ids_to_delete:
        vectorstore.delete(ids=ids_to_delete)

    ids = [doc.metadata["chunk_id"] for doc in documents]
    vectorstore.add_documents(documents=documents, ids=ids)
