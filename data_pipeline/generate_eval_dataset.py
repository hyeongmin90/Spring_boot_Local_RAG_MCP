import os
import sys
import json
import concurrent.futures
from tqdm import tqdm

# 부모 디렉토리를 경로에 추가하여 모듈을 임포트할 수 있게 함
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_pipeline.evaluate_rag import get_random_chunks, generate_questions

def generate_and_save_dataset(num_samples=100, output_file="evaluation_dataset.json"):
    """
    Randomly samples chunks and uses an LLM to generate evaluation questions, 
    saving the result to a JSON file.
    """
    print(f"=== 평가용 데이터셋 생성 시작 ({num_samples} 샘플) ===")
    
    chunks = get_random_chunks(n=num_samples)
    if not chunks:
        print("청크를 가져오지 못했습니다.")
        return

    dataset = []

    def generate_for_chunk(item):
        content = item['content']
        questions = generate_questions(content)
        return item, questions

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_chunk = {executor.submit(generate_for_chunk, item): item for item in chunks}

        for future in tqdm(concurrent.futures.as_completed(future_to_chunk), total=len(chunks), desc="LLM Question Generation"):
            try:
                item, questions = future.result()

                dataset.append({
                    "id": item['metadata'].get("chunk_id") or item.get("id"),
                    "source": item['metadata'].get("source"),
                    "content": item['content'],
                    "questions": questions
                })
            except Exception as e:
                print(f"질문 생성 중 오류 발생: {e}")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"\n데이터셋 생성이 완료되었습니다. '{output_file}'에 {len(dataset)}개의 항목이 저장되었습니다.")

if __name__ == "__main__":
    generate_and_save_dataset(num_samples=50)
