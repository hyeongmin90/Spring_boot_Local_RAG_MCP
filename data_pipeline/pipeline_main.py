import sys
import os
from dotenv import load_dotenv

# Add project root to sys.path to ensure module imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_pipeline.crawler import fetch_spring_boot_docs
from data_pipeline.processor import split_text_with_llm
from data_pipeline.storage import add_documents

def run_pipeline(url="https://docs.spring.io/spring-boot/reference/"):
    load_dotenv()
    
    print("=== Starting RAG Data Pipeline (LLM Semantic Chunking) ===")
    
    # Limit max pages for testing
    max_pages = 5
    print(f"Limiting to {max_pages} pages for testing...")
    
    # Directory for chunk logs
    log_dir = "llm_chunk_logs"
    os.makedirs(log_dir, exist_ok=True)
    
    for page in fetch_spring_boot_docs(url, max_pages=max_pages):
        url_link = page['url']
        content = page['content']
        
        # 2. Process
        print(f"  - Splitting with LLM (gpt-5-mini)...")
        chunks = split_text_with_llm(content)
        
        # Add metadata & Log to file
        log_filename = url_link.split('/')[-1]
        if not log_filename or log_filename.endswith('/'):
             path_parts = url_link.rstrip('/').split('/')
             log_filename = path_parts[-1] if path_parts else "index"
        
        # Sanitize filename
        log_filename = "".join([c for c in log_filename if c.isalpha() or c.isdigit() or c in (' ', '.', '_', '-')]).rstrip()
        log_path = os.path.join(log_dir, f"{log_filename}_chunks.txt")
        
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(f"Source: {url_link}\n")
            f.write(f"Total Chunks: {len(chunks)}\n")
            f.write("="*50 + "\n\n")
            
            for i, chunk in enumerate(chunks):
                chunk.metadata["source"] = url_link
                
                # Write to log
                f.write(f"=== Chunk {i+1} ===\n")
                f.write(f"[Summary]\n{chunk.page_content}\n\n")
                f.write(f"[Original Content]\n{chunk.metadata.get('original_content', '')}\n")
                f.write("-" * 50 + "\n\n")
                
        print(f"  - Processed: {len(chunks)} chunks from {url_link}")
        print(f"  - Log saved to: {log_path}")

        # 3. Store
        add_documents(chunks)

    print("\n=== Pipeline Completed Successfully ===")

if __name__ == "__main__":
    run_pipeline()
