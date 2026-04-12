"""
📍 1단계: API 게이트웨이 & 시맨틱 캐싱
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Logic:
  모든 질의는 FastAPI를 거쳐 Redis에서 1차 필터링됩니다.
Action:
  1. 사용자 질문을 e5-small-v2 모델로 임베딩.
  2. Redis Stack의 Vector Search를 이용해 유사도 0.90 이상의 과거 질문이 있는지 확인.
  3. 적중 시 즉시 답변 반환 (LLM 비용 0원, 지연시간 0.1초 미만).
"""

import json
import hashlib
import time
import numpy as np
from typing import Optional, Tuple

from rag_config import (
    REDIS_URL,
    CACHE_SIMILARITY_THRESHOLD,
    CACHE_INDEX_NAME,
    CACHE_KEY_PREFIX,
    EMBEDDING_DIMENSION,
)


class SemanticCache:
    """
    Redis Stack 기반 시맨틱 캐시.
    질의 임베딩의 코사인 유사도를 이용해 과거 동일/유사 질문의 답변을 즉시 반환합니다.
    """

    def __init__(self, embedding_fn=None):
        """
        Args:
            embedding_fn: 문자열 → numpy 배열(float32) 변환 함수.
                          None이면 Mock 모드로 동작합니다.
        """
        self.embedding_fn = embedding_fn
        self._redis = None
        self._index_created = False
        self._mock_store = {}  # Mock 모드용 인메모리 저장소

        try:
            import redis
            self._redis = redis.Redis.from_url(REDIS_URL, decode_responses=False)
            self._redis.ping()
            self._ensure_index()
        except Exception:
            print("[SemanticCache] Redis 연결 불가 — Mock 인메모리 캐시로 대체합니다.")
            self._redis = None

    # ──────────────────────────────────────────
    # Redis Vector Index 생성
    # ──────────────────────────────────────────
    def _ensure_index(self):
        """Redis에 벡터 인덱스가 없으면 생성합니다."""
        if self._redis is None or self._index_created:
            return

        from redis.commands.search.field import VectorField, TextField
        from redis.commands.search.indexDefinition import IndexDefinition, IndexType

        try:
            self._redis.ft(CACHE_INDEX_NAME).info()
            self._index_created = True
        except Exception:
            schema = (
                TextField("$.query", as_name="query"),
                TextField("$.answer", as_name="answer"),
                VectorField(
                    "$.embedding",
                    "FLAT",
                    {
                        "TYPE": "FLOAT32",
                        "DIM": EMBEDDING_DIMENSION,
                        "DISTANCE_METRIC": "COSINE",
                    },
                    as_name="embedding",
                ),
            )
            definition = IndexDefinition(
                prefix=[CACHE_KEY_PREFIX], index_type=IndexType.JSON
            )
            self._redis.ft(CACHE_INDEX_NAME).create_index(
                fields=schema, definition=definition
            )
            self._index_created = True
            print(f"[SemanticCache] Redis 인덱스 '{CACHE_INDEX_NAME}' 생성 완료.")

    # ──────────────────────────────────────────
    # 캐시 조회
    # ──────────────────────────────────────────
    def lookup(self, query: str) -> Optional[Tuple[str, float]]:
        """
        유사도 >= CACHE_SIMILARITY_THRESHOLD 인 캐시 항목이 있으면 (answer, score) 반환.
        없으면 None.
        """
        if self.embedding_fn is None:
            return self._mock_lookup(query)

        query_vec = self.embedding_fn(query)

        if self._redis is not None:
            return self._redis_lookup(query_vec)
        else:
            return self._mock_lookup(query)

    def _redis_lookup(self, query_vec: np.ndarray) -> Optional[Tuple[str, float]]:
        from redis.commands.search.query import Query

        q = (
            Query(f"*=>[KNN 1 @embedding $vec AS score]")
            .sort_by("score")
            .return_fields("query", "answer", "score")
            .dialect(2)
        )
        params = {"vec": query_vec.astype(np.float32).tobytes()}
        results = self._redis.ft(CACHE_INDEX_NAME).search(q, query_params=params)

        if results.total > 0:
            doc = results.docs[0]
            # Redis COSINE distance: 0 = 동일, 2 = 반대
            cosine_distance = float(doc.score)
            similarity = 1.0 - cosine_distance
            if similarity >= CACHE_SIMILARITY_THRESHOLD:
                return (doc.answer, similarity)
        return None

    def _mock_lookup(self, query: str) -> Optional[Tuple[str, float]]:
        """Mock: 정확히 같은 질문이면 적중."""
        key = self._hash(query)
        if key in self._mock_store:
            return (self._mock_store[key]["answer"], 1.0)
        return None

    # ──────────────────────────────────────────
    # 캐시 저장
    # ──────────────────────────────────────────
    def store(self, query: str, answer: str):
        """질의-답변 쌍을 캐시에 저장합니다."""
        if self.embedding_fn is None:
            self._mock_store[self._hash(query)] = {"query": query, "answer": answer}
            print(f"[SemanticCache] Mock 캐시에 저장 완료.")
            return

        query_vec = self.embedding_fn(query)

        if self._redis is not None:
            import redis
            key = f"{CACHE_KEY_PREFIX}{self._hash(query)}"
            payload = {
                "query": query,
                "answer": answer,
                "embedding": query_vec.astype(np.float32).tolist(),
            }
            self._redis.json().set(key, "$", payload)
            print(f"[SemanticCache] Redis 캐시에 저장 완료.")
        else:
            self._mock_store[self._hash(query)] = {"query": query, "answer": answer}
            print(f"[SemanticCache] Mock 캐시에 저장 완료.")

    # ──────────────────────────────────────────
    # 유틸리티
    # ──────────────────────────────────────────
    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()[:16]
