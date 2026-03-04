import sys
import os
import asyncio
import hashlib
from dotenv import load_dotenv

# Ensure the root project directory is in the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from final_pipeline.crawler import fetch_docs
from final_pipeline.processor import chunk_markdown_content
from final_pipeline.storage import add_documents

async def process_page(sem, page, category):
    """
    Async task to process a single page: Parsing -> Chunking -> Storage
    """
    url_link = page['url']
    content = page['content'] # Markdown format
    
    async with sem:
        print(f"[Process] Started chunking for: {url_link}")
        try:
            # Semantic Markdown Chunking
            chunks = chunk_markdown_content(content)
        except Exception as e:
            print(f"[Error] Failed to chunk {url_link}: {e}")
            return

        # Add global metadata & ID
        for i, chunk in enumerate(chunks):
            chunk.metadata["source"] = url_link
            chunk.metadata["category"] = category
            chunk.metadata["chunk_id"] = hashlib.md5(f"{url_link}#{i}".encode()).hexdigest()
            
        print(f"[Process] Created {len(chunks)} chunks from {url_link}")

        # Ingest to ChromaDB
        if chunks:
            await asyncio.to_thread(add_documents, chunks, "spring_docs")

async def run_ingestion_pipeline(url: str, category: str, max_pages: int = None):
    load_dotenv()
    print(f"=== Starting Final RAG Ingestion Pipeline ({category}) ===")
    
    sem = asyncio.Semaphore(5)
    tasks = []

    print("Fetching documents from crawler...")
    for page in fetch_docs(url, max_pages=max_pages):
        task = asyncio.create_task(process_page(sem, page, category))
        tasks.append(task)
        
    if tasks:
        print(f"\nScheduled {len(tasks)} tasks. Awaiting completion...")
        await asyncio.gather(*tasks)
    else:
        print("No pages found or crawled.")

    print("\n=== Ingestion Pipeline Completed ===")

if __name__ == "__main__":
    url = "https://docs.spring.io/spring-data/redis/reference/"
    #url = "https://docs.spring.io/spring-boot/reference/"
    category = "spring-data-redis"
    #category = "spring-boot"
    asyncio.run(run_ingestion_pipeline(url, category)) 
