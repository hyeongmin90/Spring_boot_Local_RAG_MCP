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

# Import storage to get VectorStore directly
from data_pipeline.storage import get_vectorstore

# 환경 변수 로드
load_dotenv()

class QAPair(BaseModel):
    question: str = Field(description="Generated question strictly based on the text")
    answer: str = Field(description="Ground truth answer to the question explicitly stated in the text")

class QAPairs(BaseModel):
    pairs: List[QAPair] = Field(description="List of QA pairs. Can be empty if the text lacks sufficient information.")

def generate_qa_pairs_from_chunk(content: str, max_pairs: int = 1) -> List[QAPair]:
    """
    LLM을 사용하여 텍스트에서 Q&A 쌍을 추출합니다.
    내용이 부실하면 적게 만들거나 안 만듭니다.
    """
    llm = ChatOpenAI(model="gpt-5-mini", temperature=0.2)

    prompt = """
    당신은 RAG 시스템 평가를 위한 데이터셋을 생성하는 전문가입니다.

    웹 페이지에서 크롤링한 마크다운 텍스트가 주어지면, 해당 텍스트를 기반으로 최대 {max_pairs}개의 질문과 답변(Q&A) 쌍을 생성하세요.

    질문과 답변은 반드시 영어로 작성해야 합니다.

    다음 지침을 반드시 따르세요.

    [문서 품질 평가]
    1. 텍스트가 너무 짧거나 실질적인 정보가 부족한 경우, 또는 메뉴/네비게이션/코드 스니펫 위주의 내용일 경우에는 질문을 적게 생성하거나 아예 생성하지 마세요(0개 가능).

    [정보 근거]
    1. 모든 답변은 반드시 제공된 텍스트 안에서 명확하게 확인할 수 있어야 합니다.
    2. 외부 지식을 사용하거나 추측하여 내용을 만들어내지 마세요.

    [질문 난이도]
    1. 질문은 텍스트 문장을 그대로 재사용하지 마세요.
    2. 가능하면 표현을 바꾸거나 의미적으로 동일한 다른 표현을 사용하세요.
    3. 질문은 텍스트의 한 문장 이상의 정보를 이해해야 답할 수 있도록 하세요.
    4. 질문은 텍스트의 핵심 개념이나 동작 원리를 묻도록 하세요.
    5. 단순한 정의 질문보다는 이유, 특징, 동작 방식 등을 묻는 질문을 우선하세요.
    6. 질문에 텍스트의 핵심 키워드를 그대로 사용하지 마세요.

    [자연스러운 질문]
    1. 실제 사용자가 검색하거나 물어볼 법한 자연스러운 질문을 작성하세요.
    2. 질문이 문서 구조를 그대로 따라가는 형태(예: "이 문서에서 ~라고 설명하는 것은 무엇인가")는 피하세요.

    [답변 작성 방식]
    1. 답변은 간결하지만 질문에 대한 핵심 정보를 충분히 포함하도록 작성하세요.
    2. 불필요하게 긴 문장이나 문단을 그대로 복사하지 마세요.

    Text:
    {text}
    """
    
    prompt_template = PromptTemplate.from_template(prompt)
    chain = prompt_template | llm.with_structured_output(QAPairs)
    
    try:
        # 텍스트가 너무 길 경우 (API 토큰 제한 및 비용 방지), 앞부분 위주로 자름 
        truncated_content = content[:15000] if len(content) > 15000 else content
        result = chain.invoke({"text": truncated_content, "max_pairs": max_pairs})
        
        # 모델이 지시를 무시하고 더 많이 생성하는 경우를 대비해 슬라이싱
        return result.pairs[:max_pairs]
    except Exception as e:
        print(f"Q&A 생성 중 오류 발생: {e}")
        return []

def create_dataset_from_crawled_md(
    collection_name: str = "spring_docs_markdown", 
    num_samples: int = 50, 
    max_pairs_per_chunk: int = 1
):
    print(f"=== VectorStore({collection_name}) 기반 LangSmith 평가용 데이터셋 구축 시작 (대상: {num_samples}개 청크) ===")
    
    # 1. DB에서 문서 가져오기
    vectorstore = get_vectorstore(collection_name)
    db_data = vectorstore.get()
    
    documents = db_data.get('documents', [])
    metadatas = db_data.get('metadatas', [])
    ids = db_data.get('ids', [])
    
    if not documents:
        print(f"오류: '{collection_name}' 컬렉션에 데이터가 없습니다. create_test_vectorstores.py를 먼저 실행하세요.")
        return
        
    print(f"VectorStore에서 총 {len(documents)}개의 청크를 발견했습니다.")

    # 랜덤 샘플링 (인덱스 추출)
    actual_samples = min(num_samples, len(documents))
    sampled_indices = random.sample(range(len(documents)), actual_samples)
    print(f"이 중 {actual_samples}개의 청크를 랜덤으로 추출하여 QA 생성을 진행합니다.")
    
    client = Client()
    
    # LangSmith Dataset 준비 (영문 이름 권장)
    dataset_name_eng = "eval_dataset_train_v3"
    try:
        dataset = client.read_dataset(dataset_name=dataset_name_eng)
        print(f"데이터셋 '{dataset_name_eng}'이(가) 이미 존재합니다. 해당 데이터셋에 추가합니다.")
    except Exception:
        # 데이터셋이 없는 경우 새로 생성
        dataset = client.create_dataset(
            dataset_name=dataset_name_eng,
            description="spring_crawled_md Markdown 청크에서 추출 및 생성된 RAG 질문-정답 데이터셋 (청크 당 최대 1개)"
        )
        print(f"LangSmith 데이터셋 '{dataset_name_eng}' 생성 완료.")

    dataset_records = []


    def process_chunk(idx):
        content = documents[idx]
        metadata = metadatas[idx] if metadatas else {}
        doc_id = ids[idx] if ids else f"chunk_{idx}"
        
        qa_pairs = generate_qa_pairs_from_chunk(content, max_pairs=max_pairs_per_chunk)
        
        all_qa_pairs = []
        for pair in qa_pairs:
            all_qa_pairs.append({
                "question": pair.question,
                "answer": pair.answer,
                "chunk_content": content,
                "source": metadata.get("source", "Unknown"),
                "header": metadata.get("header", "Unknown"),
                "chunk_id": doc_id
            })
            
        return all_qa_pairs

    # 병렬 처리로 속도 향상
    print(f"LLM을 사용하여 Q&A 쌍을 평가 및 생성하는 중... (청크당 최대 {max_pairs_per_chunk}개)")
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_chunk = {executor.submit(process_chunk, idx): idx for idx in sampled_indices}

        for future in tqdm(concurrent.futures.as_completed(future_to_chunk), total=len(sampled_indices), desc="Generating Q&A"):
            try:
                qa_data = future.result()
                if not qa_data:
                    continue
                
                for item in qa_data:
                    dataset_records.append({
                        "question": item["question"],
                        "expected_answer": item["answer"],
                        "context": item["chunk_content"],
                        "source": f"{item['source']}"
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
    # 데이터셋 구성 인자: num_samples=50 (50개 청크 랜덤추출), max_pairs_per_chunk=1
    create_dataset_from_crawled_md(num_samples=50, max_pairs_per_chunk=1)
