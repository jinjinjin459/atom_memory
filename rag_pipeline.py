"""
🛠️ Qdrant 기반 단순화된 하이브리드 RAG — 전체 파이프라인 오케스트레이터
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1단계 → 시맨틱 캐시 조회
2단계 → 쿼리 최적화 (SLM)
3단계 → Qdrant 하이브리드 검색
4단계 → 리랭킹 & 생성 (LLM)

모든 계층은 외부 서비스 미연결 시 Mock 모드로 자동 폴백됩니다.
"""

import time
from typing import List, Dict, Any, Optional, Generator

from semantic_cache import SemanticCache
from query_optimizer import QueryOptimizer
from qdrant_search import QdrantHybridSearch
from rerank_generate import Reranker, GeminiGenerator


class HybridRAGPipeline:
    """
    4단계 하이브리드 RAG 파이프라인.

    [흐름]
    질의 → Redis 캐시 체크 → SLM 쿼리 최적화 → Qdrant 하이브리드 검색
         → bge-reranker 리랭킹 → Gemini 1.5 Flash 답변 생성 → 캐시 저장
    """

    def __init__(self, use_mock: bool = True, embedding_fn=None, sparse_fn=None):
        """
        Args:
            use_mock: 전체 Mock 모드 플래그
            embedding_fn: 텍스트 → dense vector 변환 함수
            sparse_fn: 텍스트 → sparse vector 변환 함수
        """
        print("=" * 70)
        print(" [Qdrant 기반 하이브리드 RAG 파이프라인 초기화]")
        print("=" * 70)

        self.cache = SemanticCache(embedding_fn=embedding_fn)
        self.optimizer = QueryOptimizer(use_mock=use_mock)
        self.search = QdrantHybridSearch(
            embedding_fn=embedding_fn,
            sparse_fn=sparse_fn,
            use_mock=use_mock,
        )
        self.reranker = Reranker(use_mock=use_mock)
        self.generator = GeminiGenerator(use_mock=use_mock)

        print("=" * 70)
        print(" [초기화 완료]")
        print("=" * 70 + "\n")

    # ──────────────────────────────────────────
    # 문서 인덱싱
    # ──────────────────────────────────────────
    def index_documents(self, documents: List[Dict[str, Any]]):
        """
        문서를 Qdrant에 인덱싱합니다.

        Args:
            documents: [{"id": str, "text": str, "metadata": dict}, ...]
        """
        self.search.index_documents(documents)

    # ──────────────────────────────────────────
    # 전체 RAG 파이프라인 실행
    # ──────────────────────────────────────────
    def query(self, user_query: str) -> Dict[str, Any]:
        """
        4단계 하이브리드 RAG 파이프라인을 실행합니다.

        Returns:
            {
                "answer": str,
                "source": "cache" | "rag",
                "latency_ms": float,
                "steps": {
                    "cache_hit": bool,
                    "optimized_query": dict | None,
                    "search_results_count": int,
                    "reranked_count": int,
                }
            }
        """
        start = time.time()
        steps = {
            "cache_hit": False,
            "optimized_query": None,
            "search_results_count": 0,
            "reranked_count": 0,
        }

        # ─── 1단계: 시맨틱 캐시 조회 ───
        print("\n📍 [1단계] 시맨틱 캐시 조회 중...")
        cache_result = self.cache.lookup(user_query)
        if cache_result is not None:
            answer, similarity = cache_result
            latency = (time.time() - start) * 1000
            steps["cache_hit"] = True
            print(f"  ✅ 캐시 적중! (유사도: {similarity:.2f}, 지연: {latency:.1f}ms)")
            return {
                "answer": answer,
                "source": "cache",
                "latency_ms": latency,
                "steps": steps,
            }
        print("  ❌ 캐시 미스 — RAG 파이프라인을 실행합니다.")

        # ─── 2단계: 쿼리 최적화 ───
        print("\n📍 [2단계] SLM 쿼리 최적화 중...")
        optimized = self.optimizer.optimize(user_query)
        steps["optimized_query"] = optimized
        keywords = optimized["keywords"]
        refined_query = optimized["refined_query"]
        print(f"  키워드: {keywords}")
        print(f"  정제된 쿼리: {refined_query}")

        # ─── 3단계: Qdrant 하이브리드 검색 ───
        print("\n📍 [3단계] Qdrant 하이브리드 검색 중 (Dense + Sparse → RRF)...")
        search_results = self.search.hybrid_search(
            refined_query=refined_query,
            keywords=keywords,
        )
        steps["search_results_count"] = len(search_results)
        print(f"  검색 결과: {len(search_results)}개 문서")

        if not search_results:
            latency = (time.time() - start) * 1000
            answer = "관련 문서를 찾을 수 없습니다."
            return {
                "answer": answer,
                "source": "rag",
                "latency_ms": latency,
                "steps": steps,
            }

        # ─── 4단계: 리랭킹 & 생성 ───
        print("\n📍 [4단계-1] bge-reranker 리랭킹 중...")
        reranked = self.reranker.rerank(user_query, search_results)
        steps["reranked_count"] = len(reranked)

        print(f"\n📍 [4단계-2] Gemini 1.5 Flash 답변 생성 중...")
        answer = self.generator.generate(user_query, reranked)

        # 답변을 캐시에 저장
        self.cache.store(user_query, answer)

        latency = (time.time() - start) * 1000
        print(f"\n  ⏱️ 전체 파이프라인 완료 (지연: {latency:.1f}ms)")

        return {
            "answer": answer,
            "source": "rag",
            "latency_ms": latency,
            "steps": steps,
        }

    # ──────────────────────────────────────────
    # 스트리밍 답변 생성
    # ──────────────────────────────────────────
    def query_stream(
        self, user_query: str
    ) -> Generator[str, None, None]:
        """
        스트리밍 방식의 답변 생성.
        FastAPI의 StreamingResponse와 결합할 수 있습니다.
        """
        # 캐시 체크
        cache_result = self.cache.lookup(user_query)
        if cache_result is not None:
            yield cache_result[0]
            return

        # 쿼리 최적화
        optimized = self.optimizer.optimize(user_query)

        # 하이브리드 검색
        search_results = self.search.hybrid_search(
            refined_query=optimized["refined_query"],
            keywords=optimized["keywords"],
        )

        if not search_results:
            yield "관련 문서를 찾을 수 없습니다."
            return

        # 리랭킹
        reranked = self.reranker.rerank(user_query, search_results)

        # 스트리밍 생성
        full_answer = []
        for chunk in self.generator.generate_stream(user_query, reranked):
            full_answer.append(chunk)
            yield chunk

        # 전체 답변 캐시 저장
        self.cache.store(user_query, "".join(full_answer))
