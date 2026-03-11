import sys
import os
import asyncio
import hashlib
from dotenv import load_dotenv
from tqdm.asyncio import tqdm

# Ensure the root project directory is in the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_pipeline.crawler import fetch_docs
from data_pipeline.processor.processor import chunk_markdown_content
from data_pipeline.storage import add_documents

async def process_page(sem, page, category):
    """
    Async task to process a single page: Parsing -> Chunking -> Storage
    """
    url_link = page['url']
    content = page['content'] # Markdown format
    
    async with sem:
        tqdm.write(f"  > [Start] Processing: {url_link}")
        try:
            # Semantic Markdown Chunking
            chunks = chunk_markdown_content(content)
        except Exception as e:
            tqdm.write(f"  ! [Error] Failed to chunk {url_link}: {e}")
            return

        # Add global metadata & ID
        for i, chunk in enumerate(chunks):
            chunk.metadata["source"] = url_link
            chunk.metadata["category"] = category
            chunk.metadata["chunk_id"] = hashlib.md5(f"{url_link}#{i}".encode()).hexdigest()
            
        tqdm.write(f"  - [Done] Created {len(chunks)} chunks from {url_link}")

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
        for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing Pages"):
            await f
    else:
        print("No pages found or crawled.")

    print("\n=== Ingestion Pipeline Completed ===")

if __name__ == "__main__":
    print("select category")
    print("1. spring-boot")
    print("2. spring-data-jpa")
    print("3. spring-data-redis")
    print("4. spring-security")
    print("5. spring-cloud-gateway")
    input_category = input("Enter category: ")
    if input_category == "1":
        url = "https://docs.spring.io/spring-boot/reference/"
        category = "spring-boot"
    elif input_category == "2":
        url = "https://docs.spring.io/spring-data/jpa/reference/"
        category = "spring-data-jpa"
    elif input_category == "3":
        url = "https://docs.spring.io/spring-data/redis/reference/"
        category = "spring-data-redis"
    elif input_category == "4":
        url = "https://docs.spring.io/spring-security/reference/"
        category = "spring-security"
    elif input_category == "5":
        url = "https://docs.spring.io/spring-cloud-gateway/reference/"
        category = "spring-cloud-gateway"
    else:
        print("Invalid category")
        sys.exit(1)
    
    print(f"Ingesting {category} docs from {url}")
    asyncio.run(run_ingestion_pipeline(url, category)) 
