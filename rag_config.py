"""
🛠️ Qdrant 기반 단순화된 하이브리드 RAG — 설정 모듈
모든 환경변수와 상수를 한 곳에서 관리합니다.
"""

import os
from dotenv import load_dotenv

load_dotenv()


# ──────────────────────────────────────────────
# 1. 임베딩 모델 (e5-small-v2)
# ──────────────────────────────────────────────
EMBEDDING_MODEL_NAME = "gemini-embedding-001"
EMBEDDING_DIMENSION = 3072  # gemini-embedding-001 출력 차원

# ──────────────────────────────────────────────
# 2. Redis Stack (시맨틱 캐시)
# ──────────────────────────────────────────────
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")
CACHE_SIMILARITY_THRESHOLD = 0.90  # 유사도 임계값
CACHE_INDEX_NAME = "rag_cache_idx"
CACHE_KEY_PREFIX = "rag:cache:"

# ──────────────────────────────────────────────
# 3. SLM (쿼리 최적화)
# ──────────────────────────────────────────────
SLM_BACKEND = os.environ.get("SLM_BACKEND", "ollama")  # "ollama" | "vllm"
SLM_MODEL = os.environ.get("SLM_MODEL", "llama3:8b")
SLM_BASE_URL = os.environ.get("SLM_BASE_URL", "http://localhost:11434")

# ──────────────────────────────────────────────
# 4. Qdrant
# ──────────────────────────────────────────────
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", None)
QDRANT_COLLECTION_NAME = os.environ.get("QDRANT_COLLECTION", "hybrid_rag_docs_v2")

# Named Vectors 설정
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"

# ──────────────────────────────────────────────
# 5. 리랭커 (bge-reranker-v2-m3)
# ──────────────────────────────────────────────
RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
RERANK_TOP_K_INPUT = 10   # 검색 후 리랭커에 투입할 문서 수
RERANK_TOP_K_OUTPUT = 5   # 리랭커가 최종 선정할 문서 수

# ──────────────────────────────────────────────
# 6. Final LLM (Gemini 1.5 Flash)
# ──────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", None)
GEMINI_MODEL = "gemini-flash-latest"

# ──────────────────────────────────────────────
# 7. HuggingFace Token (리랭커 Inference API 용)
# ──────────────────────────────────────────────
HF_TOKEN = os.environ.get("HF_TOKEN", None)
