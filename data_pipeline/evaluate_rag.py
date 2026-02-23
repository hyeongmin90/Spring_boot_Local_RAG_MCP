import os
import sys
import random
from tqdm import tqdm
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List
from datetime import datetime

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from data_pipeline.storage import get_vectorstore, query_documents, mmr_query_documents, query_hybrid

# Load environment variables (OPENAI_API_KEY)
load_dotenv()

def get_random_chunks(n=10):
    """
    Fetches random chunks from the vector store.
    """
    vectorstore = get_vectorstore()
    
    # Get all documents from Chroma
    # By default, vectorstore.get() returns all documents if no filter is applied
    print("Fetching documents from vector store...")
    db_data = vectorstore.get()
    
    if not db_data or 'documents' not in db_data or not db_data['documents']:
        print("No documents found in the vector store.")
        return []
        
    total_docs = len(db_data['documents'])
    print(f"Total documents available: {total_docs}")
    
    # Randomly sample N indices
    sample_size = min(n, total_docs)
    sampled_indices = random.sample(range(total_docs), sample_size)
    
    sampled_chunks = []
    for idx in sampled_indices:
        doc_content = db_data['documents'][idx]
        metadata = db_data['metadatas'][idx] if 'metadatas' in db_data and db_data['metadatas'] else {}
        doc_id = db_data['ids'][idx] if 'ids' in db_data and db_data['ids'] else None
        
        # Only select chunks that have a reasonable length to generate a good question
        if len(doc_content) > 100:
            sampled_chunks.append({
                "content": doc_content,
                "metadata": metadata,
                "id": doc_id
            })
            
    print(f"Sampled {len(sampled_chunks)} viable chunks for evaluation.")
    return sampled_chunks

class Questions(BaseModel):
    questions: List[str]

def generate_questions(chunk_content):
    """
    Uses LLM to generate 3 challenging questions based on the provided text chunk.
    """
    llm = ChatOpenAI(model="gpt-5-mini", temperature=0.5)

    prompt = """
    You are an expert evaluator for a RAG system.
    Given the following text, generate exactly 3 distinct questions that can be answered strictly and explicitly using only the provided text.

    Follow these rules strictly:
    1. Do not copy exact sentences from the text.
    2. Include a small number of important keywords from the text.
    3. The answer to each question must be clearly and explicitly stated in the text.
    4. Do NOT require external knowledge, architectural reasoning, performance analysis, or unstated implications.
    5. Vary the difficulty by increasing reasoning complexity within the text only.
    6. Output only the questions.

    Text:
    {text}

    Questions:
    """
    
    prompt = PromptTemplate.from_template(prompt)
    
    chain = prompt | llm.with_structured_output(Questions)
    questions = chain.invoke({"text": chunk_content})

    return questions.questions

def evaluate_retrieval(question, expected_id, expected_source, method="dense", k=10):
    """
    Queries the vector store with the generated question and checks if the expected chunk is retrieved.
    Calculates the rank of the expected chunk.
    """
    if method == "mmr":
        results = mmr_query_documents(question, k=k)
    elif method == "hybrid":
        results = query_hybrid(question, k=k)
    else:
        results = query_documents(question, k=k)

    for rank, doc in enumerate(results, start=1):
        retrieved_id = getattr(doc, 'id', None) or doc.metadata.get("chunk_id")
        retrieved_source = doc.metadata.get("source")
        
        # Check if it matches the expected chunk
        # Match only by chunk ID for strict evaluation
        if expected_id and retrieved_id == expected_id:
            return rank
            
    return -1 # Not found within top k

import concurrent.futures

def run_evaluation(num_samples=10, max_k=10):
    print("=== Starting RAG Quantitative Evaluation ===")
    
    chunks = get_random_chunks(n=num_samples)
    if not chunks:
        return
        
    print("\nGenerating questions concurrently...")
    
    # Process question generation concurrently
    def generate_for_chunk(item):
        content = item['content']
        questions = generate_questions(content)
        return item, questions

    chunk_questions_map = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        # Submit all tasks
        future_to_chunk = {executor.submit(generate_for_chunk, item): item for item in chunks}
        
        # Gather results with progress bar
        for future in tqdm(concurrent.futures.as_completed(future_to_chunk), total=len(chunks), desc="LLM Generation"):
            try:
                item, questions = future.result()
                chunk_questions_map.append((item, questions))
            except Exception as exc:
                print(f"\nChunk generation generated an exception: {exc}")

    print("\nEvaluating retrieval...")
    
    methods = ["dense", "hybrid"]
    all_metrics = {m: {
        "hits_5": 0, "mrr_sum_5": 0.0, 
        "hits_10": 0, "mrr_sum_10": 0.0, 
        "hits_max_k": 0, "mrr_sum_max_k": 0.0, 
        "results_log": []
    } for m in methods}
    
    total_questions = 0
    
    for i, (item, questions) in enumerate(tqdm(chunk_questions_map, desc="Retrieval Testing")):
        metadata = item['metadata']
        
        expected_chunk_id = metadata.get("chunk_id")
        expected_source = metadata.get("source", "Unknown")
        
        for q_idx, question in enumerate(questions):
            total_questions += 1
            
            for method in methods:
                # 2. Evaluate Retrieval (Fetch up to max_k, e.g., 50)
                rank = evaluate_retrieval(question, expected_chunk_id, expected_source, method=method, k=max_k)
                
                # 3. Calculate Metrics
                # --- Top 5 Metrics ---
                is_hit_5 = 0 < rank <= 5
                if is_hit_5:
                    all_metrics[method]["hits_5"] += 1
                    all_metrics[method]["mrr_sum_5"] += 1.0 / rank

                # --- Top 10 Metrics ---
                is_hit_10 = 0 < rank <= 10
                if is_hit_10:
                    all_metrics[method]["hits_10"] += 1
                    all_metrics[method]["mrr_sum_10"] += 1.0 / rank

                # --- Top max_k Metrics ---
                is_hit_max_k = 0 < rank <= max_k
                if is_hit_max_k:
                    all_metrics[method]["hits_max_k"] += 1
                    all_metrics[method]["mrr_sum_max_k"] += 1.0 / rank
                    
                # Log result for this question
                all_metrics[method]["results_log"].append({
                    "chunk_idx": i + 1,
                    "q_idx": q_idx + 1,
                    "question": question,
                    "expected_source": expected_source,
                    "rank": rank,
                })
        
    for method in methods:
        # Final Metrics calculation
        tq = total_questions if total_questions > 0 else 1
    
        hit_rate_5 = all_metrics[method]["hits_5"] / tq
        mrr_5 = all_metrics[method]["mrr_sum_5"] / tq
        
        hit_rate_10 = all_metrics[method]["hits_10"] / tq
        mrr_10 = all_metrics[method]["mrr_sum_10"] / tq
    
        hit_rate_max_k = all_metrics[method]["hits_max_k"] / tq
        mrr_max_k = all_metrics[method]["mrr_sum_max_k"] / tq
        
        print("\n" + "="*50)
        print(f"=== Evaluation Results ({method.upper()}) ===")
        print(f"Total Chunks Sampled: {len(chunks)}")
        print(f"Total Questions Evaluated: {total_questions}")
        print("-" * 25)
        print("--- Top 5 Metrics ---")
        print(f"Top-5 Hit Rate: {hit_rate_5:.2%} ({all_metrics[method]['hits_5']}/{total_questions})")
        print(f"Top-5 MRR: {mrr_5:.4f}")
        print("-" * 25)
        print("--- Top 10 Metrics ---")
        print(f"Top-10 Hit Rate: {hit_rate_10:.2%} ({all_metrics[method]['hits_10']}/{total_questions})")
        print(f"Top-10 MRR: {mrr_10:.4f}")
        print("-" * 25)
        print(f"--- Top {max_k} Metrics ---")
        print(f"Top-{max_k} Hit Rate: {hit_rate_max_k:.2%} ({all_metrics[method]['hits_max_k']}/{total_questions})")
        print(f"Top-{max_k} MRR: {mrr_max_k:.4f}")
        print("="*50)
    
        log_file = f"evaluation_log_{method}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(f"Retrieval Method: {method.upper()}\n\n")
            f.write("=== Evaluation Results ===\n")
            f.write(f"Total Chunks Sampled: {len(chunks)}\n")
            f.write(f"Total Questions Evaluated: {total_questions}\n")
            f.write("-" * 25 + "\n")
            f.write("--- Top 5 Metrics ---\n")
            f.write(f"Top-5 Hit Rate: {hit_rate_5:.2%} ({all_metrics[method]['hits_5']}/{total_questions})\n")
            f.write(f"Top-5 MRR: {mrr_5:.4f}\n")
            f.write("-" * 25 + "\n")
            f.write("--- Top 10 Metrics ---\n")
            f.write(f"Top-10 Hit Rate: {hit_rate_10:.2%} ({all_metrics[method]['hits_10']}/{total_questions})\n")
            f.write(f"Top-10 MRR: {mrr_10:.4f}\n")
            f.write("-" * 25 + "\n")
            f.write(f"--- Top {max_k} Metrics ---\n")
            f.write(f"Top-{max_k} Hit Rate: {hit_rate_max_k:.2%} ({all_metrics[method]['hits_max_k']}/{total_questions})\n")
            f.write(f"Top-{max_k} MRR: {mrr_max_k:.4f}\n")
            f.write("="*50 + "\n\n")
            f.write("--- Detailed Log ---\n")
            for i, log in enumerate(all_metrics[method]["results_log"], 1):
                if log['rank'] > 0:
                    status = f"✅ HIT (Rank: {log['rank']})"
                else:
                    status = f"❌ MISS (Not in top-{max_k})"
                    
                f.write(f"Q{i}: {log['question']}\n")
                f.write(f"   Source: {log['expected_source']}\n")
                f.write(f"   Result: {status}\n\n")

if __name__ == "__main__":
    # You can adjust the number of samples and max_k value here
    run_evaluation(num_samples=50, max_k=50)
