# LangChain RAG & Code Helper Agent

> **Note**: 이 프로젝트는 **LangChain 프레임워크 기반의 고급 RAG (Retrieval-Augmented Generation) 파이프라인 구축 및 평가**를 핵심 목표로 하는 학습 및 실험용 프로젝트입니다. 
> 부가적으로 로컬 파일 시스템 제어 및 자동화를 돕는 **코드 에이전트(DevAgent)** 기능이 보조 도구로 포함되어 있습니다.

## 프로젝트 개요
이 프로젝트의 핵심은 최신 프레임워크 문서와 웹 데이터를 크롤링하여 고성능 벡터 데이터베이스에 저장하고, 사용자의 질문에 대해 가장 연관성 높은 문서를 정확하게 찾아내는 **고급 검색(Retrieval) 시스템 구축과 그 성능을 정량적으로 평가하는 프레임워크**입니다.

보조 기능인 `DevAgent`는 사용자의 자연어 요청을 해석하여 파일 읽기/쓰기, 터미널 명령어 실행 등 로컬 개발 보조와 RAG 테스트 자동화를 돕는 서브 어시스턴트 역할을 수행합니다.

## 프로젝트 목표
코드 에이전트를 사용한 개발자동화 도구 구축 및 RAG를 이용한 최신 프레임워크 대응

## 사용 기술
- Langchain
- ChromaDB
- Cohere Reranker
- LangSmith
- OpenAI GPT-5-mini
- MCP (Model Context Protocol) / FastMCP

## 핵심 기능 (Core Features)

### 1. RAG 파이프라인
- Spring boot docs를 RAG로 구축한 후 Langchain을 이용하여 챗봇을 구축
- Spring boot, Spring data redis의 docs를 대상으로 진행

*   **하이브리드 검색 (Hybrid Search)**: ChromaDB를 이용한 Dense(의미 기반) 검색과 BM25를 이용한 Sparse(키워드 기반) 검색을 결합하여 검색 정확도 극대화.
*   **Cohere Reranker 통합**: 1차로 검색된 수십 개의 후보군 문서를 딥러닝 문장 비교 모델(Cohere API)을 통해 2차 정밀 재배열(Reranking)하여 최상위 정답(Top-1, Top-5) 적중률을 비약적으로 상승.
*   **다양한 데이터 처리 프로세서**: 웹 크롤러 통합, 마크다운/HTML 파싱, 청킹(Chunking) 및 메타데이터 주입 후 로컬 벡터 DB 관리.

### 2. 시스템 자동 평가 및 시각화 프레임워크 (RAG Evaluation)
*   **LangSmith 연동**: "LLM as a judge" 개념을 도입, GPT-5-mini 모델이 RAG 파이프라인의 답변 정확성(Correctness), 근거성(Groundedness), 검색 적합성(Retrieval relevance) 여부를 직접 평가하고 LangSmith 대시보드에 추적.
*   **Retriever 자체 정량 평가**: Top-1/Top-5/Top-10 Hit Rate(적중률), MRR(Mean Reciprocal Rank), 어휘적/의미적 중복도(Redundancy) 등 필수 검색 성능 지표들을 벤치마킹.
*   **차트 시각화**: `matplotlib`을 이용해 여러 검색 기법(Dense 단독 vs Hybrid vs Hybrid+Reranker) 간의 성능 지표 차이를 막대그래프 이미지로 자동 출력 및 저장 (`results/` 폴더).

### 3. MCP 서버 (RAG as a Tool)
*   **FastMCP** 기반의 MCP(Model Context Protocol) 서버로 RAG 검색 기능을 외부 AI 도구에서 직접 호출할 수 있도록 노출.
*   `get_docs(query)` 툴 하나로 Hybrid Search(Dense + BM25) 결과를 JSON 형태로 반환. (reranker 제외)

### 4. 코드 에이전트 (DevAgent) - 보조 기능
*   RAG 파이프라인 테스트 및 파일 수정, 환경 설정 등의 로컬 반복 작업을 돕기 위해 설계된 메인-서브 계층형 에이전트.
*   터미널 명령어 실행, 파일 코드 리뷰 및 수정 기능 수행.

---

## 최종 채택된 방법론 (Final Chosen Methods)
수많은 실험과 성능 평가를 거쳐, 본 프로젝트에서 최종적으로 채택한 각 단계별 최적의 파이프라인 구성은 다음과 같습니다.

1. **문서 파싱 (Markdown Parsing)**
   * **도구**: `BeautifulSoup` + `markdownify`
   * **전략**: 복잡한 HTML 구조를 그대로 두지 않고, 기술 문서의 의미론적 구조(Semantic Structure)와 코드 블록을 보존하기 위해 문서를 깔끔한 Markdown 포맷으로 우선 정규화합니다.
2. **청킹 (Chunking Strategy)**
   * **도구**: `MarkdownHeaderTextSplitter` + `RecursiveCharacterTextSplitter`
   * **전략**: 코드가 많이 포함된 공식 문서의 특성상 Header(#, ##) 단위로 1차 분할하여 문맥(Context) 유실을 방지합니다. 그 후에도 너무 긴 단락은 2차로 Character 레벨에서 분할하는 정밀한 하이브리드 청킹 기법을 사용합니다.
3. **검색 및 리트리버 (Retriever)**
   * **구조**: `Hybrid Search` (Chroma Dense 0.7 : BM25 Sparse 0.3) + `Cohere Reranker`
   * **전략**: 임베딩 기반의 의미(Vector) 검색과 키워드 기반의 검색을 7:3 가중치(RRF 알고리즘)로 결합해 1차 대규모 후보군(최소 30개 이상)을 추출합니다. 이후 문맥 파악에 특화된 딥러닝 모델인 **Cohere Reranker**를 통과시켜 최종 순위를 재조정함으로써 검색 적중률(Top-1/Top-5 Hit Rate)을 극대화시켰습니다.
4. **RAG 파이프라인 접근법 (Pipeline Architecture)**
   * **구조**: `Simple RAG Chain`
   * **전략**: 복잡한 LangGraph 기반 Tool 호출 구조 대신, 검색 결과를 프롬프트에 직접 컨텍스트(Context)로 주입하는 표준 RAG 방식을 채택했습니다. 
   * **결과**: LLM의 도구 판단 여부를 거치지 않아 지연 시간(Latency) 최소화에 기여하며, `ask_query` 함수 등의 모듈화를 통해 외부(코드 보조 에이전트 등)에서 직접 문자열 답변만 받아볼 수 있게 단순화했습니다.

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
*   **최종 파이프라인 문서 수집 및 저장 (Ingest)**
    ```bash
    python final_pipeline/main_ingest.py
    ```
*   **최종 파이프라인 RAG 챗봇 실행 (단순형)**
    ```bash
    python final_pipeline/agent.py
    ```
*   **RAG 종합 성능 평가 (Hit Rate, MRR 계산 및 시각화 차트 생성)**
    ```bash
    python data_pipeline/evaluation/evaluate_retriever_comprehensive.py
    ```
*   **LangSmith LLM-as-a-judge 평가 실행**
    연동된 LangSmith 대시보드에서 평가 결과를 확인할 수 있습니다.
    데이터셋을 미리 등록해야 합니다.
    ```bash
    python data_pipeline/evaluation/run_langsmith_eval.py
    ```
*   **MCP 서버 직접 실행** (디버그/독립 실행용)
    ```bash
    python mcp/server.py
    ```
    MCP 클라이언트(Antigravity 등)에서 자동 기동하려면 `mcp_config.json`에 등록합니다:
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
*   **보조 코드 에이전트 실행**
    ```bash
    python main.py
    # 또는 Windows 배치 파일 사용: ai.bat
    ```

---

## 디렉토리 구조 (Architecture)

```text
langchainDev/
├── final_pipeline/                  # 최종 정제된 RAG 파이프라인 모듈
│   ├── crawler.py                   # 웹 HTML ➔ 시맨틱 Markdown 파싱
│   ├── processor.py                 # Markdown 하이브리드 청킹
│   ├── storage.py                   # ChromaDB 세션 및 중복 제어
│   ├── retriever.py                 # Hybrid (Dense + BM25) + Cohere Reranker
│   ├── agent.py                     # Simple RAG Chain 구조 챗봇
│   └── main_ingest.py               # 파이프라인 최종 수집/구축 엔트리포인트 스크립트
│
├── data_pipeline/                   # 기존 RAG 엔진 보관 / 평가 실험실
│   ├── evaluation/                  # RAG 성능 평가 및 시각화 모듈 (HitRate, MRR 등)
│   ├── processor/                   # 기본 및 마크다운 테스트를 위한 분할/처리 함수
│   └── pipeline_main.py             # 실험 파이프라인 엔트리 스크립트
│
├── results/                         # 평가 결과 기록 보관소 (자동 생성)
│   ├── mmr_eval                     # MMR 평가 결과
│   ├── hybrid_eval                  # Hybrid 평가 결과
│   ├── comprehensive_eval_summary_*.txt    # 통합 분석 지표 로그
│   └── comprehensive_eval_chart_*.png      # Matplotlib 성능 비교 시각화 그래프
│
├── dataset/                         # 데이터셋
│   ├── dev.json                     # RAG 데이터셋
│   └── test.json                    # RAG 데이터셋
│
├── chroma_db/                       # 로컬 벡터 데이터베이스(Chroma)
│
├── mcp/                             # MCP 서버 (RAG as a Tool)
│   └── server.py                    # FastMCP 기반 get_docs 툴 노출
│
├── agent/                           # 보조 코드 에이전트 패키지 (터미널, 파일 제어 로직)
│   ├── sub_agent.py / tools.py      # 내부 도구 및 서브 에이전트 구현
│   └── ui.py / utils.py             # 스트리밍 및 상태 표시 UI
│
├── main.py                          # 보조 에이전트 실행 진입점
├── requirements.txt                 # 전체 프로젝트 패키지 의존성 명세
└── README.md                        # 프로젝트 가이드
```

## 참고 사항
*   본 프로젝트는 로컬 Windows 환경 및 Python 3.12+ 버전 위에서 테스트되었습니다.
*   **저장소 트래픽 초과 주의**: Cohere Reranker 사용 시 짧은 시간에 대량의 평가(`evaluate_retriever_comprehensive.py` 등)를 진행할 경우 Trial API Key의 Rate Limit(분당 요청 횟수)에 주의하세요. (Production Key 권장)
