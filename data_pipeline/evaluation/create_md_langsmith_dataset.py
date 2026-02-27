import os
import sys
import random
import concurrent.futures
from tqdm import tqdm
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List

# 부모 디렉토리를 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from langsmith import Client
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

# 환경 변수 로드
load_dotenv()

class QAPair(BaseModel):
    question: str = Field(description="Generated question strictly based on the text")
    answer: str = Field(description="Ground truth answer to the question explicitly stated in the text")

class QAPairs(BaseModel):
    pairs: List[QAPair] = Field(description="List of QA pairs. Can be empty if the text lacks sufficient information.")

def generate_qa_pairs_from_md(content: str, max_pairs: int = 3) -> List[QAPair]:
    """
    LLM을 사용하여 텍스트에서 Q&A 쌍을 추출합니다.
    내용이 부실하면 적게 만들거나 안 만듭니다.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

    prompt = """
    You are an expert dataset creator for a RAG evaluation system.
    Given the following markdown text formatted from a crawled web page, generate up to {max_pairs} Question and Answer pairs.
    
    CRITICAL INSTRUCTIONS:
    1. Assess the content quality. If the text is too short, lacks substantial factual information, is mostly boilerplate, code snippets without context, or navigation menus, you MUST generate FEWER pairs (e.g., 1 or 2) or an EMPTY list of pairs (0 pairs).
    2. Only generate questions if there is clear, explicitly stated factual information to form a reliable answer.
    3. The answer must be explicitly found in the text. Do NOT use outside knowledge or hallucinate details.
    4. Provide concise but complete answers.
    5. The question should be realistic for a user to ask.

    Text:
    {text}
    """
    
    prompt_template = PromptTemplate.from_template(prompt)
    chain = prompt_template | llm.with_structured_output(QAPairs)
    
    try:
        # 텍스트가 너무 길 경우 (API 토큰 제한 및 비용 방지), 앞부분 위주로 자름 
        # (원한다면 다른 로직을 추가할 수 있지만 일반적으로 15000자면 충분)
        truncated_content = content[:15000] if len(content) > 15000 else content
        result = chain.invoke({"text": truncated_content, "max_pairs": max_pairs})
        
        # 모델이 지시를 무시하고 더 많이 생성하는 경우를 대비해 슬라이싱
        return result.pairs[:max_pairs]
    except Exception as e:
        print(f"Q&A 생성 중 오류 발생: {e}")
        return []

def create_dataset_from_crawled_md(
    md_dir: str = "../../spring_crawled_md", 
    dataset_name: str = "RAG_평가_데이터셋_MD", # 원한다면 영문으로 변경 가능: "rag-eval-from-md-dataset"
    num_samples: int = 50, 
    max_pairs_per_page: int = 3
):
    print(f"=== 원본 MD 기반 LangSmith 평가용 데이터셋 구축 시작 (대상: {num_samples}개 파일) ===")
    
    abs_md_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), md_dir))
    if not os.path.exists(abs_md_dir):
        print(f"오류: 경로를 찾을 수 없습니다 -> {abs_md_dir}")
        return

    # md 파일 목록 가져오기
    all_md_files = [f for f in os.listdir(abs_md_dir) if f.endswith('.md')]
    if not all_md_files:
        print(f"오류: {abs_md_dir} 에 마크다운 파일이 없습니다.")
        return

    # 랜덤 샘플링
    actual_samples = min(num_samples, len(all_md_files))
    sample_files = random.sample(all_md_files, actual_samples)
    print(f"총 {len(all_md_files)}개의 마크다운 파일 중 {actual_samples}개를 랜덤으로 추출했습니다.")

    client = Client()
    
    # LangSmith Dataset 준비 (영문 이름 권장)
    dataset_name_eng = "rag-eval-from-md-dataset"
    try:
        dataset = client.read_dataset(dataset_name=dataset_name_eng)
        print(f"데이터셋 '{dataset_name_eng}'이(가) 이미 존재합니다. 해당 데이터셋에 추가합니다.")
    except Exception:
        # 데이터셋이 없는 경우 새로 생성
        dataset = client.create_dataset(
            dataset_name=dataset_name_eng,
            description="spring_crawled_md 원본 페이지에서 추출 및 생성된 RAG 질문-정답 데이터셋 (페이지당 최대 3개, 부실하면 적게 생성)"
        )
        print(f"LangSmith 데이터셋 '{dataset_name_eng}' 생성 완료.")

    dataset_records = []

    def process_file(filename):
        filepath = os.path.join(abs_md_dir, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            
        qa_pairs = generate_qa_pairs_from_md(content, max_pairs=max_pairs_per_page)
        return filename, content, qa_pairs

    # 병렬 처리로 속도 향상
    print(f"LLM을 사용하여 Q&A 쌍을 평가 및 생성하는 중... (페이지당 최대 {max_pairs_per_page}개)")
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_file = {executor.submit(process_file, f): f for f in sample_files}

        for future in tqdm(concurrent.futures.as_completed(future_to_file), total=len(sample_files), desc="Generating Q&A"):
            try:
                filename, content, qa_pairs = future.result()
                if not qa_pairs:
                    continue
                
                for pair in qa_pairs:
                    dataset_records.append({
                        "question": pair.question,
                        "expected_answer": pair.answer,
                        "context": content,
                        "source": filename
                    })
            except Exception as e:
                print(f"병렬 처리 중 예외 발생: {e}")

    if not dataset_records:
        print("경고: 생성된 Q&A 쌍이 하나도 없습니다. 문서 내용이 모두 부실하거나 오류가 발생했을 수 있습니다.")
        return

    print(f"\n총 {len(dataset_records)}개의 Q&A 쌍이 성공적으로 생성되었습니다.")
    print("LangSmith에 업로드를 시작합니다...")
    
    for record in tqdm(dataset_records, desc="Uploading to LangSmith"):
        try:
            client.create_example(
                inputs={
                    "question": record["question"]
                },
                outputs={
                    "answer": record["expected_answer"],
                    "context": record["context"], # 정답 비교 및 검증에 원본 컨텍스트 사용 가능
                    "source": record["source"]
                },
                dataset_name=dataset_name_eng
            )
        except Exception as e:
            print(f"업로드 중 오류 발생: {e}")
            
    print(f"\n✅ 데이터셋 구축이 완료되었습니다!")
    print(f"👉 LangSmith Dashboard에서 '{dataset_name_eng}' 데이터셋을 확인하세요.")

if __name__ == "__main__":
    # 데이터셋 구성 인자: num_samples=50 (50개 랜덤추출), max_pairs_per_page=3 (최대 3개 질문)
    create_dataset_from_crawled_md(num_samples=50, max_pairs_per_page=3)
