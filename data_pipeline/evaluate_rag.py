import os
import sys
import random
from tqdm import tqdm
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from data_pipeline.storage import get_vectorstore, query_documents

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
        
        # Only select chunks that have a reasonable length to generate a good question
        if len(doc_content) > 100:
            sampled_chunks.append({
                "content": doc_content,
                "metadata": metadata
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
    
    prompt = PromptTemplate.from_template(
        "You are an expert evaluator for a RAG system.\n"
        "Given the following text, generate exactly 3 distinct questions that can be answered by reading this text.\n"
        "Follow these rules strictly:\n"
        "1. Do not use the exact phrasing from the text.\n"
        "2. Do not copy key keywords directly from the text.\n"
        "3. At least one question must be highly abstract or conceptual.\n"
        "4. The questions must sound like natural, realistic inquiries a developer might ask.\n"
        "5. You may use terminology not present in the text, as long as the underlying meaning/answer remains the same.\n"
        "6. Three questions should each have a different level of difficulty.\n"
        "7. Output only the questions, nothing else.\n"
        "Text:\n{text}\n\n"
        "Questions"
    )
    
    chain = prompt | llm.with_structured_output(Questions)
    questions = chain.invoke({"text": chunk_content})

    return questions.questions

def evaluate_retrieval(question, expected_chunk_id, expected_source, k=10):
    """
    Queries the vector store with the generated question and checks if the expected chunk is retrieved.
    Calculates the rank of the expected chunk.
    """
    # Fetch top k results
    # Adjust category if necessary, or just search globally
    results = query_documents(question, k=k)
    
    for rank, doc in enumerate(results, start=1):
        retrieved_chunk_id = doc.metadata.get("chunk_id")
        retrieved_source = doc.metadata.get("source")
        
        # Check if it matches the expected chunk
        # Matching by chunk_id is preferred if available, otherwise by source as a loose fallback
        if expected_chunk_id and retrieved_chunk_id == expected_chunk_id:
            return rank
        elif not expected_chunk_id and expected_source and retrieved_source == expected_source:
             return rank
            
    return -1 # Not found within top k

import concurrent.futures

def run_evaluation(num_samples=10, max_k=10):
    print("=== Starting RAG Quantitative Evaluation ===")
    
    chunks = get_random_chunks(n=num_samples)
    if not chunks:
        return
        
    results_log = []
    
    # Track metrics for top-5
    hits_5 = 0
    mrr_sum_5 = 0.0
    
    # Track metrics for top-10
    hits_10 = 0
    mrr_sum_10 = 0.0

    # Track metrics for top-max_k
    hits_max_k = 0
    mrr_sum_max_k = 0.0
    
    total_questions = 0
    
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
    
    for i, (item, questions) in enumerate(tqdm(chunk_questions_map, desc="Retrieval Testing")):
        metadata = item['metadata']
        
        expected_chunk_id = metadata.get("chunk_id")
        expected_source = metadata.get("source", "Unknown")
        
        for q_idx, question in enumerate(questions):
            total_questions += 1
            
            # 2. Evaluate Retrieval (Fetch up to max_k, e.g., 10)
            rank = evaluate_retrieval(question, expected_chunk_id, expected_source, k=max_k)
            
            # 3. Calculate Metrics
            # --- Top 10 Metrics ---
            is_hit_10 = 0 < rank <= 10
            if is_hit_10:
                hits_10 += 1
                mrr_sum_10 += 1.0 / rank
                
            # --- Top 5 Metrics ---
            is_hit_5 = 0 < rank <= 5
            if is_hit_5:
                hits_5 += 1
                mrr_sum_5 += 1.0 / rank

            is_hit_max_k = 0 < rank <= max_k
            if is_hit_max_k:
                hits_max_k += 1
                mrr_sum_max_k += 1.0 / rank
                
            # Log result for this question
            results_log.append({
                "chunk_idx": i + 1,
                "q_idx": q_idx + 1,
                "question": question,
                "expected_source": expected_source,
                "rank": rank,
                "is_hit_5": is_hit_5,
                "is_hit_10": is_hit_10,
                "is_hit_max_k": is_hit_max_k
            })
        
    # Final Metrics calculation
    total_questions = total_questions if total_questions > 0 else 0

    hit_rate_5 = hits_5 / total_questions
    mrr_5 = mrr_sum_5 / total_questions
    
    hit_rate_10 = hits_10 / total_questions
    mrr_10 = mrr_sum_10 / total_questions

    hit_rate_max_k = hits_max_k / total_questions
    mrr_max_k = mrr_sum_max_k / total_questions
    
    print("\n" + "="*50)
    print("=== Evaluation Results ===")
    print(f"Total Chunks Sampled: {len(chunks)}")
    print(f"Total Questions Evaluated: {total_questions}")
    print("-" * 25)
    print("--- Top 5 Metrics ---")
    print(f"Top-5 Hit Rate: {hit_rate_5:.2%} ({hits_5}/{total_questions})")
    print(f"Top-5 MRR: {mrr_5:.4f}")
    print("-" * 25)
    print("--- Top 10 Metrics ---")
    print(f"Top-10 Hit Rate: {hit_rate_10:.2%} ({hits_10}/{total_questions})")
    print(f"Top-10 MRR: {mrr_10:.4f}")
    print("-" * 25)
    print(f"--- Top {max_k} Metrics ---")
    print(f"Top-{max_k} Hit Rate: {hit_rate_max_k:.2%} ({hits_max_k}/{total_questions})")
    print(f"Top-{max_k} MRR: {mrr_max_k:.4f}")
    print("="*50 + "\n")

    
    # Detailed log
    # print("--- Detailed Log ---")
    # for i, log in enumerate(results_log, 1):
    #     if log['rank'] > 0:
    #         status = f"✅ HIT (Rank: {log['rank']})"
    #     else:
    #         status = f"❌ MISS (Not in top-{max_k})"
            
    #     print(f"Q{i}: {log['question']}")
    #     print(f"   Source: {log['expected_source']}")
    #     print(f"   Result: {status}\n")

    with open("evaluation_log.txt", "w", encoding="utf-8") as f:
        f.write("=== Evaluation Results ===\n")
        f.write(f"Total Chunks Sampled: {len(chunks)}\n")
        f.write(f"Total Questions Evaluated: {total_questions}\n")
        f.write("-" * 25 + "\n")
        f.write("--- Top 5 Metrics ---\n")
        f.write(f"Top-5 Hit Rate: {hit_rate_5:.2%} ({hits_5}/{total_questions})\n")
        f.write(f"Top-5 MRR: {mrr_5:.4f}\n")
        f.write("-" * 25 + "\n")
        f.write("--- Top 10 Metrics ---\n")
        f.write(f"Top-10 Hit Rate: {hit_rate_10:.2%} ({hits_10}/{total_questions})\n")
        f.write(f"Top-10 MRR: {mrr_10:.4f}\n")
        f.write("-" * 25 + "\n")
        f.write(f"--- Top {max_k} Metrics ---\n")
        f.write(f"Top-{max_k} Hit Rate: {hit_rate_max_k:.2%} ({hits_max_k}/{total_questions})\n")
        f.write(f"Top-{max_k} MRR: {mrr_max_k:.4f}\n")
        f.write("="*50 + "\n\n")
        f.write("--- Detailed Log ---\n")
        for i, log in enumerate(results_log, 1):
            if log['rank'] > 0:
                status = f"✅ HIT (Rank: {log['rank']})"
            else:
                status = f"❌ MISS (Not in top-{max_k})"
                
            f.write(f"Q{i}: {log['question']}\n")
            f.write(f"   Source: {log['expected_source']}\n")
            f.write(f"   Result: {status}\n\n")

if __name__ == "__main__":
    # You can adjust the number of samples and max_k value here
    run_evaluation(num_samples=100, max_k=50)
