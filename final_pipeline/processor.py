from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_markdown_content(markdown_text: str, max_chunk_size: int = 1000, chunk_overlap: int = 200) -> list[Document]:
    """
    Chunks markdown text based on headers (#, ##) and size limits.
    Uses RecursiveCharacterTextSplitter for sections under a header that are too long.
    
    Returns:
        List[Document]: List of chunked documents with 'header' metadata.
    """
    if not markdown_text:
        return []
        
    documents = []
    lines = markdown_text.split('\n')
    
    current_chunk_lines = []
    current_header = "Introduction"
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    
    def flush():
        nonlocal current_chunk_lines
        if not current_chunk_lines:
            return
            
        text = "\n".join(current_chunk_lines).strip()
        if not text:
            return
            
        splits = text_splitter.split_text(text)
        
        for split in splits:
            doc = Document(
                page_content=split,
                metadata={"header": current_header}
            )
            documents.append(doc)
            
        current_chunk_lines = []
        
    in_code_block = False
    
    for line in lines:
        if line.strip().startswith("```"):
            in_code_block = not in_code_block
            
        is_header = False
        header_level = 0
        
        if not in_code_block:
            if line.startswith("# "):
                is_header = True
                header_level = 1
            elif line.startswith("## "):
                is_header = True
                header_level = 2
            elif line.startswith("###"):
                is_header = True
                header_level = 3
        
        if is_header and header_level <= 2:
            flush()
            current_header = line.lstrip('#').strip()
            current_chunk_lines.append(line)
        else:
            current_chunk_lines.append(line)
            
    flush()
    return documents
