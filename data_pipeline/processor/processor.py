from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


#basic split text
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


#markdown chunking
def chunk_markdown_content(markdown_text, max_chunk_size=1000):
    """
    Chunks markdown text based on headers (#, ##) and size limits.
    
    Strategy:
    1. Parse the markdown to identify headers.
    2. Split content by Level 1 (#) and Level 2 (##) headers.
    3. Treat Level 3+ (###) as content within the parent chunk, unless it triggers a size split.
    4. Ensure chunks don't break code blocks.
    
    Returns:
        List[Document]: List of chunked documents with metadata.
    """
    if not markdown_text:
        return []
        
    documents = []
    lines = markdown_text.split('\n')
    
    current_chunk_lines = []
    current_chunk_size = 0
    current_header = "Introduction"
    
    # Helper to flush current chunk
    def flush():
        nonlocal current_chunk_lines, current_chunk_size
        if not current_chunk_lines:
            return
            
        # Join lines
        text = "\n".join(current_chunk_lines).strip()
        if not text:
            return
            
        # Metadata
        doc = Document(
            page_content=text,
            metadata={
                "header": current_header
            }
        )
        documents.append(doc)
        
        current_chunk_lines = []
        current_chunk_size = 0
        
    in_code_block = False
    
    for line in lines:
        # Toggle code block status
        if line.strip().startswith("```"):
            in_code_block = not in_code_block
            
        # Check for headers (only if not in code block)
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
        
        # Decide whether to split
        # Split on H1 and H2
        if is_header and header_level <= 2:
            flush() # Finish previous chunk
            
            # Start new chunk with this header
            # Remove MD header tokens for metadata display (optional)
            current_header = line.lstrip('#').strip()
            
            # Add header line to content as well? Yes, to keep context.
            current_chunk_lines.append(line)
            current_chunk_size += len(line)
            
        else:
            # Content or Minor Header
            line_len = len(line) + 1 # +1 for newline
            
            # Check size limit (Soft split)
            # We try to split on paragraphs or minor headers if possible, 
            # but for simplicity here we just check if adding this line would strictly overflow 
            # AND we are not in a code block.
            
            # Better logic: If size > max, flush at next header or paragraph break?
            # Basic logic: If current size > max, we SHOULD split. 
            # But we don't want to split inside a sentence or code block.
            
            if current_chunk_size + line_len > max_chunk_size and not in_code_block:
                # If we are at a minor header, existing split point
                if is_header and header_level >= 3:
                     flush()
                     current_chunk_lines.append(line)
                     current_chunk_size += len(line)
                     continue
                
                # If just text, we flush to enforce size limit.
                # Ideally we check for paragraph breaks (empty lines), 
                # but splitting at line boundary is acceptable for now.
                flush()
            
            current_chunk_lines.append(line)
            current_chunk_size += line_len
            
    # Final flush
    flush()
    
    return documents


    

    
