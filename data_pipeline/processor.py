from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

def split_text(text, chunk_size=1000, chunk_overlap=200):
    """
    Splits text into chunks using RecursiveCharacterTextSplitter with custom logic.
    If text length is less than 1.5 * chunk_size, returns a single chunk.
    Otherwise, splits with specified overlap.
    Returns a list of Document objects.
    """
    if not text:
        return []
        
    # Custom logic: if text is short enough, keep it whole
    if len(text) < (chunk_size * 1.5):
        return [Document(page_content=text)]
        
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.create_documents([text])

def enhance_chunk_with_llm(chunk_text, llm=None):
    """
    Uses LLM to generate a concise summary or keywords for the chunk.
    This metadata can be stored with the chunk for better retrieval.
    """
    if not llm:
        llm = ChatOpenAI(model="gpt-5-mini", temperature=0)

    prompt = PromptTemplate.from_template(
        "Summarize the key technical concepts in the following text in 1 sentence:\n\n{text}"
    )
    chain = prompt | llm
    try:
        summary = chain.invoke({"text": chunk_text})
        return summary.content
    except Exception as e:
        print(f"Error enhancing chunk: {e}")
        return ""

def process_documents(documents, use_llm=False):
    """
    Iterates through documents and optionally adds LLM-generated metadata.
    """
    if use_llm:
        print(f"Enhancing {len(documents)} chunks with LLM...")
        llm = ChatOpenAI(model="gpt-5-mini", temperature=0)
        for i, doc in enumerate(documents):
            print(f"Processing chunk {i+1}/{len(documents)}...")
            summary = enhance_chunk_with_llm(doc.page_content, llm)
            doc.metadata["summary"] = summary
    return documents
