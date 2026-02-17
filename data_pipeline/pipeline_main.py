import sys
import os
from dotenv import load_dotenv

# Add project root to sys.path to ensure module imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_pipeline.crawler import fetch_spring_boot_docs
from data_pipeline.processor import split_text
from data_pipeline.storage import add_documents

def run_pipeline(url="https://docs.spring.io/spring-boot/reference/"):
    load_dotenv()
    
    print("=== Starting RAG Data Pipeline ===")
    
    for page in fetch_spring_boot_docs(url):
        url_link = page['url']
        content = page['content']
        
        # 2. Process
        chunks = split_text(content, chunk_size=1000, chunk_overlap=200)
        
        # Add metadata
        for chunk in chunks:
            chunk.metadata["source"] = url_link
            
        print(f"  - Processed: {len(chunks)} chunks from {url_link}")

        # 3. Store
        add_documents(chunks)

    print("\n=== Pipeline Completed Successfully ===")

if __name__ == "__main__":
    run_pipeline()
