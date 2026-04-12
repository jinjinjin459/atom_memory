"""
🌐 FastAPI 기반 RAG API 게이트웨이
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
모든 질의는 이 FastAPI 서버를 거쳐 처리됩니다.

실행:
  uvicorn rag_api:app --reload --port 8000

엔드포인트:
  POST /query         — 동기 답변
  POST /query/stream  — 스트리밍 답변
  POST /index         — 문서 인덱싱
  GET  /health        — 헬스 체크
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from rag_pipeline import HybridRAGPipeline

app = FastAPI(
    title="Qdrant 하이브리드 RAG API",
    description="시맨틱 캐싱 + SLM 쿼리 최적화 + Qdrant 하이브리드 검색 + 리랭킹 + Gemini 생성",
    version="1.0.0",
)

# ──────────────────────────────────────────
# 파이프라인 인스턴스 (앱 시작 시 초기화)
# ──────────────────────────────────────────
pipeline: Optional[HybridRAGPipeline] = None


@app.on_event("startup")
async def startup():
    global pipeline
    pipeline = HybridRAGPipeline(use_mock=True)


# ──────────────────────────────────────────
# 요청/응답 모델
# ──────────────────────────────────────────
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    source: str  # "cache" | "rag"
    latency_ms: float
    steps: Dict[str, Any]

class IndexRequest(BaseModel):
    documents: List[Dict[str, Any]]  # [{"id": str, "text": str, "metadata": dict}]


# ──────────────────────────────────────────
# 엔드포인트
# ──────────────────────────────────────────
@app.get("/health")
async def health_check():
    return {"status": "ok", "pipeline": pipeline is not None}


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """동기식 RAG 질의 처리."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="파이프라인이 초기화되지 않았습니다.")
    result = pipeline.query(request.query)
    return QueryResponse(**result)


@app.post("/query/stream")
async def query_stream_endpoint(request: QueryRequest):
    """스트리밍 RAG 질의 처리."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="파이프라인이 초기화되지 않았습니다.")

    def stream_generator():
        for chunk in pipeline.query_stream(request.query):
            yield chunk

    return StreamingResponse(
        stream_generator(),
        media_type="text/plain; charset=utf-8",
    )


@app.post("/index")
async def index_endpoint(request: IndexRequest):
    """문서 인덱싱."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="파이프라인이 초기화되지 않았습니다.")
    pipeline.index_documents(request.documents)
    return {"status": "indexed", "count": len(request.documents)}
