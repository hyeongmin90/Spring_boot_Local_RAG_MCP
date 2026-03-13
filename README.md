# LangChain RAG & LangGraph Agent

> **Note**: 이 프로젝트는 **LangChain / LangGraph 프레임워크 기반의 고급 RAG (Retrieval-Augmented Generation) 파이프라인 구축 및 평가**를 핵심 목표로 하는 학습 및 실험용 프로젝트입니다.
> Spring Framework 공식 문서를 벡터 데이터베이스에 구축하고, LangGraph 기반의 RAG 에이전트로 정확한 답변을 생성합니다.

## 프로젝트 개요
이 프로젝트의 핵심은 최신 프레임워크 문서와 웹 데이터를 크롤링하여 고성능 벡터 데이터베이스에 저장하고, 사용자의 질문에 대해 가장 연관성 높은 문서를 정확하게 찾아내는 **고급 검색(Retrieval) 시스템 구축과 그 성능을 정량적으로 평가하는 프레임워크**입니다.

**LangGraph RAG 에이전트**는 질문 분석 → 쿼리 재작성 → 하이브리드 검색 → 답변 생성의 명시적인 플로우로 구성되어 있으며, 각 단계를 독립적인 노드로 관리합니다.

## 프로젝트 목표
RAG를 이용한 최신 프레임워크 문서 검색 시스템 구축 및 성능 평가

## 사용 기술
- LangChain / LangGraph
- ChromaDB
- Cohere Reranker
- LangSmith
- OpenAI GPT-5-mini
- MCP / FastMCP

## 핵심 기능 (Core Features)

### 1. RAG 파이프라인
- Spring boot docs를 RAG로 구축한 후 Langchain을 이용하여 챗봇을 구축
  - Spring boot
  - Spring data jpa
  - Spring data redis
  - Spring security
  - Spring cloud gateway

*   **하이브리드 검색 (Hybrid Search)**: ChromaDB를 이용한 Dense(의미 기반) 검색과 BM25를 이용한 Sparse(키워드 기반) 검색을 결합하여 검색 정확도 극대화.
*   **Cohere Reranker 통합**: 1차로 검색된 수십 개의 후보군 문서를 딥러닝 문장 비교 모델(Cohere API)을 통해 2차 정밀 재배열(Reranking)하여 최상위 정답(Top-1, Top-5) 적중률을 비약적으로 상승.
*   **다양한 데이터 처리 프로세서**: 웹 크롤러 통합, 마크다운/HTML 파싱, 청킹(Chunking) 및 메타데이터 주입 후 로컬 벡터 DB 관리.

### 2. 시스템 자동 평가 및 시각화 프레임워크 (RAG Evaluation)
*   **LangSmith 연동**: "LLM as a judge" 개념을 도입, GPT-5-mini 모델이 RAG 파이프라인의 답변 정확성(Correctness), 근거성(Groundedness), 검색 적합성(Retrieval relevance) 여부를 직접 평가하고 LangSmith 대시보드에 추적.
*   **Retriever 자체 정량 평가**: Top-1/Top-5/Top-10 Hit Rate(적중률), MRR(Mean Reciprocal Rank), 어휘적/의미적 중복도(Redundancy) 등 필수 검색 성능 지표들을 벤치마킹.
*   **차트 시각화**: `matplotlib`을 이용해 여러 검색 기법(Dense 단독 vs Hybrid vs Hybrid+Reranker) 간의 성능 지표 차이를 막대그래프 이미지로 자동 출력 및 저장 (`results/` 폴더).

### 3. MCP 서버 (RAG as a Tool)
*   **FastMCP** 기반의 MCP(Model Context Protocol) 서버로 RAG 검색 기능을 외부 AI 도구에서 직접 호출할 수 있도록 노출.
*   `get_docs(query)` 툴 하나로 Hybrid Search(Dense + BM25) 결과를 JSON 형태로 반환. 
*   `get_docs_with_reranker(query)` 툴 하나로 Hybrid Search(Dense + BM25) 결과를 JSON 형태로 반환. (Cohere Reranker 통합)

### 4. LangGraph RAG 에이전트 (메인 기능)
*   Spring Framework 공식 문서에 대한 질문에 최적화된 **Self-Correction 기반 에이전트**.
*   LangGraph `StateGraph`로 4개의 독립 노드(retrieve → grade_docs → rewrite → generate)를 조합하여 검색 결과의 품질을 스스로 평가하고 필요시 쿼리를 재작성합니다.
*   Pydantic `with_structured_output`으로 각 노드의 출력 타입을 보장.
*   CLI 루프 형태로 대화형 사용 가능.

---

## 최종 채택된 방법론 (Final Chosen Methods)
수많은 실험과 성능 평가를 거쳐, 본 프로젝트에서 최종적으로 채택한 각 단계별 최적의 파이프라인 구성은 다음과 같습니다.

1. **문서 파싱**
   * **도구**: `BeautifulSoup` + `markdownify`
   * **전략**: 복잡한 HTML 구조를 그대로 두지 않고, 기술 문서의 의미론적 구조(Semantic Structure)와 코드 블록을 보존하기 위해 문서를 깔끔한 Markdown 포맷으로 우선 정규화합니다.
2. **청킹**
   * **도구**: `MarkdownHeaderTextSplitter` + `RecursiveCharacterTextSplitter`
   * **전략**: 코드가 많이 포함된 공식 문서의 특성상 Header(#, ##) 단위로 1차 분할하여 문맥(Context) 유실을 방지합니다. 그 후에도 너무 긴 단락은 2차로 Character 레벨에서 분할하는 정밀한 하이브리드 청킹 기법을 사용합니다.
3. **검색 및 리트리버**
   * **구조**: `Hybrid Search` (Chroma Dense 0.7 : BM25 Sparse 0.3) + `Cohere Reranker`
   * **전략**: 임베딩 기반의 의미(Vector) 검색과 키워드 기반의 검색을 7:3 가중치(RRF 알고리즘)로 결합해 1차 대규모 후보군(최소 30개 이상)을 추출합니다. 이후 문맥 파악에 특화된 딥러닝 모델인 **Cohere Reranker**를 통과시켜 최종 순위를 재조정함으로써 검색 적중률(Top-1/Top-5 Hit Rate)을 극대화시켰습니다.
4. **RAG 에이전트**
   * **구조**: `LangGraph StateGraph` (Self-Correction RAG 플로우)
   * **플로우**:
     ```mermaid
     graph TD
       START((START)) --> Retrieve[retrieve]
       Retrieve --> Grade[grade_docs]
       Grade -- "재작성 필요" --> Rewrite[rewrite]
       Rewrite --> Retrieve
       Grade -- "충분함" --> Generate[generate]
       Generate --> END((END))
     ```
   * **retrieve 노드**: 원본 질문 또는 재작성된 쿼리로 Hybrid Search + Reranker 수행
   * **grade_docs 노드**: 검색된 문서들이 질문에 답변하기에 충분한지 LLM이 평가
   * **rewrite 노드**: 검색 품질이 낮을 경우, 질문을 분석하여 카테고리를 추론하고 영어 기술 쿼리로 재작성
   * **generate 노드**: 검색 문서 기반 한국어 최종 답변 생성
5. **MCP 서버**
   * **구조**: `FastMCP`
   * **전략**: MCP(Model Context Protocol)를 이용하여 RAG 검색 기능을 외부 AI 도구에서 직접 호출할 수 있도록 노출합니다.
   * **특징**: 빠른 일반 검색(`get_docs`)과 고성능 재배열 검색(`get_docs_with_reranker`) 2가지 툴을 제공합니다. Reranker 툴은 무료 API 키 사용시 한도로 제한적일 수 있습니다.
---

## 🛠 설치 및 실행 (Installation & Setup)

### 1. 환경 설정
프로젝트 루트에 `.env` 파일을 생성하고 필수 API 키를 추가합니다:
```txt
# RAG 임베딩, 평가 프롬프트, 에이전트 구동용
OPENAI_API_KEY="YOUR_OPENAI_API_KEY"

# Reranker 성능 최적화용
COHERE_API_KEY="YOUR_COHERE_API_KEY"

# RAG 평가 트래킹 및 모니터링
LANGCHAIN_TRACING_V2="true"
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
LANGCHAIN_API_KEY="YOUR_LANGSMITH_API_KEY"
LANGCHAIN_PROJECT="YOUR_PROJECT_NAME"
```

### 2. 의존성 패키지 설치
최신 Python 환경(가상환경 권장)에서 필요 라이브러리들을 설치합니다:
```bash
pip install -r requirements.txt
```

### 3. 주요 스크립트 실행
*   **최종 파이프라인 문서 수집 및 저장**
    > **Note**: 프로젝트 초기 복제 시 로컬에 벡터 DB(ChromaDB)가 구성되어 있지 않습니다. RAG 검색 기능과 챗봇이 작동하려면 **반드시 가장 먼저 이 스크립트를 한 번 실행하여 공식 문서를 수집하고 데이터베이스를 채워야 합니다.**
    ```bash
    python final_pipeline/main_ingest.py
    ```

*   **최종 파이프라인 RAG 챗봇 실행**
    ```bash
    python final_pipeline/agent.py
    ```
*   **Retriever 종합 성능 평가 (Hit Rate, MRR 계산 및 시각화 차트 생성)**
    ```bash
    python data_pipeline/evaluation/evaluate_retriever_comprehensive.py
    ```
    > **평가 결과 예시**  
    > ![RAG 검색 성능 평가 차트](results/hybrid_eval/5_docs/test_result/comprehensive_eval_chart_20260311_191037.png)

    - hybrid retriever 평가 결과
    - 5개 docs의 청크를 기반으로 질문 생성
    - 평가 방법
      - 임의의 청크 수집하여 LLM을 활용해 질문 생성
      - 생성된 질문으로 원본 청크를 top-k개로 가져오는지 retriever 평가
      - 일반화, 추상화, 의미적 재구성, 다중 정보 결합 등의 지시로 질문 난이도를 높여 Recall을 낮춤
      - 평가 지표: Hit Rate, MRR
    
*   **LangSmith LLM-as-a-judge 평가 실행**
    ```bash
    python data_pipeline/evaluation/run_langsmith_eval.py
    ```
    LangSmith 대시보드에서 평가 결과를 확인할 수 있습니다.
    데이터셋을 미리 등록해야 합니다.


*   **MCP 서버 직접 실행** (디버그/독립 실행용)
    ```bash
    python mcp/server.py
    ```
    MCP 클라이언트(Antigravity, Cursor 등)에서 자동 기동하려면 `mcp_config.json`에 등록합니다:
    ```json
    {
      "mcpServers": {
        "rag-server": {
          "command": "<venv경로>/python.exe",
          "args": ["<프로젝트루트>/mcp/server.py"]
        }
      }
    }
    ```
*   **LangGraph RAG 에이전트 실행 (메인)**
    ```bash
    python main.py
    # 또는 Windows 배치 파일 사용
    ai.bat
    ```
    실행 시 아래와 같이 단계별 진행 상황이 출력됩니다:
    ```
    User: security jwt 설정 방법은?

      ▶ 분석: 카테고리=Spring Security  |  재작성=필요
      ▶ 쿼리 재작성: Spring Security JWT token authentication configuration
      ▶ 검색 완료: 5개 문서 검색됨

    Agent:
    Spring Security에서 JWT를 설정하려면...
    ```

---

## 디렉토리 구조 (Architecture)

```text
langchainDev/
├── agent/                           # LangGraph RAG 에이전트 패키지 (메인)
│   ├── state.py                     # AgentState TypedDict 정의
│   ├── prompts.py                   # 각 노드용 프롬프트 및 카테고리 목록
│   ├── nodes.py                     # 4개 노드 함수 구현 (analyze/rewrite/retrieve/generate)
│   ├── graph.py                     # LangGraph StateGraph 조립 및 컴파일
│   └── __init__.py                  # build_graph() 노출
│
├── pipeline/                        # RAG 엔진 및 평가 실험실
│   ├── crawler/                     # 웹 크롤러
│   ├── storage.py                   # ChromaDB CRUD
│   ├── retriever.py                 # Retriever 구현
│   ├── ingest.py                    # 문서 수집/벡터DB 구축 엔트리포인트
│   ├── evaluation/                  # RAG 성능 평가 및 시각화 모듈
│   └── processor/                   # 청킹/처리 함수
│
├── legacy/                          # 구버전 코드 에이전트 보관
│
├── results/                         # 평가 결과 기록 보관소 (자동 생성)
│
├── chroma_db/                       # 로컬 벡터 데이터베이스(Chroma)
│
├── mcp/                             # MCP 서버 (RAG as a Tool)
│   └── server.py                    # FastMCP 기반 get_docs 툴 노출
│
├── main.py                          # LangGraph RAG 에이전트 CLI 진입점
├── ai.bat                           # RAG 에이전트 Windows 배치 실행 파일
├── requirements.txt                 # 전체 프로젝트 패키지 의존성 명세
└── README.md                        # 프로젝트 가이드
```

## 참고 사항
*   본 프로젝트는 로컬 Windows 환경 및 Python 3.12+ 버전 위에서 테스트되었습니다.
*   **저장소 트래픽 초과 주의**: Cohere Reranker 사용 시 짧은 시간에 대량의 평가(`evaluate_retriever_comprehensive.py` 등)를 진행할 경우 Trial API Key의 Rate Limit(분당 요청 횟수)에 주의하세요. (Production Key 권장)

---

## 라이선스 (License)

이 프로젝트는 **MIT 라이선스(MIT License)**를 따릅니다.
자세한 내용은 [LICENSE](LICENSE) 파일을 확인하세요. 누구나 자유롭게 사용, 수정, 배포 및 상업적 목적으로 활용할 수 있습니다.
