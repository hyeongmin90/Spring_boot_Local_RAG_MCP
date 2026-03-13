from langchain_core.prompts import ChatPromptTemplate

# 지원 카테고리 목록
SUPPORTED_CATEGORIES = [
    "spring-boot",
    "spring-data-jpa",
    "spring-data-redis",
    "spring-security",
    "spring-cloud-gateway",
]

# ──────────────────────────────────────────────
# 1. Analyze Prompt
#    - 질문 이해 및 카테고리 추론
#    - 쿼리 재작성 필요 여부 판단
# ──────────────────────────────────────────────
ANALYZE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a query analysis assistant for a Spring Framework documentation RAG system.\n"
        "Given a user's question, determine:\n"
        "1. Whether the query needs to be rewritten for better search performance.\n"
        "   - Rewrite if: the query is vague, colloquial, too short, uses abbreviations, or mixes Korean/English poorly.\n"
        "   - Do NOT rewrite if: the query is already clear, specific, and well-formed for technical search.\n"
        "2. The most relevant documentation category from the list below.\n"
        "   - Return null if no specific category is clearly implied.\n\n"
        "Supported categories:\n"
        "{categories}"
    ),
    ("human", "{question}"),
])

# ──────────────────────────────────────────────
# 2. Rewrite Prompt
#    - 검색에 최적화된 영어 쿼리로 변환
# ──────────────────────────────────────────────
REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a query optimization assistant for a technical documentation search engine.\n"
        "Rewrite the given user question into a concise, English search query optimized for semantic search.\n"
        "Rules:\n"
        "- Keep it under 20 words.\n"
        "- Use technical terminology from Spring Framework.\n"
        "- If a category is provided, incorporate it to make the query more specific.\n"
        "Category (may be null): {category}"
    ),
    ("human", "{question}"),
])

# ──────────────────────────────────────────────
# 3. Generate Prompt
#    - 검색 결과를 바탕으로 최종 답변 생성
# ──────────────────────────────────────────────
GENERATE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful Spring Framework expert assistant.\n"
        "Use ONLY the following retrieved context to answer the user's question.\n"
        "If the context does not contain enough information, say you don't know.\n"
        "Provide clear, code-centric answers where applicable.\n"
        "Always answer in Korean.\n\n"
        "Context:\n{context}"
    ),
    ("human", "{question}"),
])
