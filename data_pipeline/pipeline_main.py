import sys
import os
import asyncio
from dotenv import load_dotenv
from tqdm.asyncio import tqdm
import hashlib

from data_pipeline.crawler import fetch_docs
from data_pipeline.processor.processor import chunk_markdown_content
from data_pipeline.storage import add_documents


async def process_page(sem, page, log_dir, category):
    """
    Async task to process a single page:
    1. Split text
    2. Save logs
    3. Store in Vector DB
    """
    url_link = page['url']
    content = page['content'] # This is now Markdown Text
    
    async with sem:
        tqdm.write(f"  > [Start] Processing: {url_link}")
        
        # 2. Process (Markdown Chunking)
        try:
            chunks = chunk_markdown_content(content)
        except Exception as e:
            tqdm.write(f"  ! [Error] Failed to chunk {url_link}: {e}")
            return

        # Add metadata & Log to file
        log_filename = url_link.split('/')[-1]
        
        # If filename is empty (e.g. url ends in /), try previous segment
        if not log_filename:
             path_parts = url_link.rstrip('/').split('/')
             log_filename = path_parts[-1] if path_parts else "index"
        elif log_filename.endswith('/'): # Safety
             path_parts = url_link.rstrip('/').split('/')
             log_filename = path_parts[-1] if path_parts else "index"
        

        for i, chunk in enumerate(chunks):
            chunk.metadata["source"] = url_link
            # Add docs_type to metadata
            chunk.metadata["category"] = category
            chunk.metadata["chunk_id"] = hashlib.md5(f"{url_link}#{i}".encode()).hexdigest()
        # # Debug: 청킹 결과 로그 저장
        # log_filename = "".join([c for c in log_filename if c.isalpha() or c.isdigit() or c in (' ', '.', '_', '-')]).rstrip()
        # log_path = os.path.join(log_dir, f"{log_filename}_chunks.txt")
        
        # with open(log_path, "w", encoding="utf-8") as f:
        #     f.write(f"Source: {url_link}\n")
        #     f.write(f"Total Chunks: {len(chunks)}\n")
        #     f.write("="*50 + "\n\n")
        #     for i, chunk in enumerate(chunks):
        #         f.write(f"=== Chunk {i+1} ===\n")
        #         f.write(f"Header: {chunk.metadata.get('header', 'N/A')}\n")
        #         f.write(f"Category: {category}\n")
        #         f.write(f"[Content]\n{chunk.page_content}\n\n")
        #         f.write("-" * 50 + "\n\n")
        
        tqdm.write(f"  - [Done] {len(chunks)} chunks from {url_link}")

        # 3. Store (Offload blocking IO to thread)
        if chunks:
            await asyncio.to_thread(add_documents, chunks)

async def run_pipeline_async(url="https://docs.spring.io/spring-boot/reference/", category="spring-boot"):
    load_dotenv()
    
    print(f"=== Starting Async RAG Data Pipeline ({category}) ===")
    
    # Directory for chunk logs
    log_dir = "chunk_logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # Limit pages for testing if needed
    max_pages = None
    print(f"Limiting to {max_pages} pages..." if max_pages else "No page limit set.")

    # Concurrency Control
    sem = asyncio.Semaphore(5)
    tasks = []

    # 1. Crawl (Generator) - Sync
    # We iterate the generator and spawn async tasks
    print("Fetching pages from crawler...")
    
    # fetch_docs uses the url passed
    for i, page in enumerate(fetch_docs(url, max_pages=max_pages)):
        task = asyncio.create_task(process_page(sem, page, log_dir, category))
        tasks.append(task)
    
    if tasks:
        print(f"\nScheduled {len(tasks)} tasks. Waiting for completion...")
        # await asyncio.gather(*tasks)
        for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing Pages"):
            await f
    else:
        print("No pages found or crawled.")

    print("\n=== Pipeline Completed ===")

if __name__ == "__main__":
    print("1. Spring Boot Docs")
    print("2. Spring Data Redis Docs")
    selection = input("select docs: ")

    url = ""
    category = ""
    
    if selection == "1":
        url = "https://docs.spring.io/spring-boot/reference/"
        category = "spring-boot"
    elif selection == "2":
        url = "https://docs.spring.io/spring-data/redis/reference/"
        category = "spring-data-redis"
    else:
        print("Invalid selection")
        sys.exit(1)

    asyncio.run(run_pipeline_async(url, category))
