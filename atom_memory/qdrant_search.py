"""
📍 3단계: Qdrant 통합 하이브리드 검색 (Single DB)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Tech: Qdrant (Cloud 또는 Docker)
Key Setup: 단일 Collection 내에 Named Vectors 설정 (Dense + Sparse).
  - Dense: refined_query 기반 (의미 검색)
  - Sparse: keywords 기반 (키워드/BM25 검색)
Action:
  - client.search_batch()를 사용하여 두 가지 검색을 동시에 수행.
  - Qdrant의 RRF(Reciprocal Rank Fusion) 기능을 호출하여 자동으로 결과 병합.
"""

import uuid
from typing import List, Dict, Any, Optional

from rag_config import (
    QDRANT_URL,
    QDRANT_API_KEY,
    QDRANT_COLLECTION_NAME,
    DENSE_VECTOR_NAME,
    SPARSE_VECTOR_NAME,
    EMBEDDING_DIMENSION,
    RERANK_TOP_K_INPUT,
)


class QdrantHybridSearch:
    """
    Qdrant 단일 Collection에서 Dense(의미) + Sparse(BM25) 검색을 동시에 수행하고
    RRF(Reciprocal Rank Fusion)로 자동 병합하는 하이브리드 검색 엔진.
    """

    def __init__(self, embedding_fn=None, sparse_fn=None, use_mock: bool = True):
        """
        Args:
            embedding_fn: 문자열 → dense vector(list[float]) 변환 함수
            sparse_fn:    문자열 → sparse vector {indices: list, values: list} 변환 함수
            use_mock:     True이면 Qdrant 없이 인메모리 검색
        """
        self.embedding_fn = embedding_fn
        self.sparse_fn = sparse_fn
        self.use_mock = use_mock
        self._client = None
        self._mock_documents = []  # Mock 저장소

        if not self.use_mock:
            try:
                from qdrant_client import QdrantClient
                self._client = QdrantClient(
                    url=QDRANT_URL,
                    api_key=QDRANT_API_KEY,
                )
                self._ensure_collection()
                print(f"[QdrantHybridSearch] Qdrant 연결 성공 ({QDRANT_URL})")
            except Exception as e:
                print(f"[QdrantHybridSearch] Qdrant 연결 실패: {e} — Mock 모드로 대체합니다.")
                self.use_mock = True

    # ──────────────────────────────────────────
    # Collection 생성
    # ──────────────────────────────────────────
    def _ensure_collection(self):
        """단일 Collection에 Named Vectors(Dense + Sparse)를 설정합니다."""
        from qdrant_client.models import (
            VectorParams,
            SparseVectorParams,
            Distance,
        )

        collections = [c.name for c in self._client.get_collections().collections]
        if QDRANT_COLLECTION_NAME not in collections:
            self._client.create_collection(
                collection_name=QDRANT_COLLECTION_NAME,
                vectors_config={
                    DENSE_VECTOR_NAME: VectorParams(
                        size=EMBEDDING_DIMENSION,
                        distance=Distance.COSINE,
                    )
                },
                sparse_vectors_config={
                    SPARSE_VECTOR_NAME: SparseVectorParams()
                },
            )
            print(f"[QdrantHybridSearch] Collection '{QDRANT_COLLECTION_NAME}' 생성 완료.")
        else:
            print(f"[QdrantHybridSearch] Collection '{QDRANT_COLLECTION_NAME}' 이미 존재.")

    # ──────────────────────────────────────────
    # 문서 인덱싱 (Upsert)
    # ──────────────────────────────────────────
    def index_documents(self, documents: List[Dict[str, Any]]):
        """
        문서를 Qdrant에 인덱싱합니다.

        Args:
            documents: [{"id": str, "text": str, "metadata": dict}, ...]
        """
        if self.use_mock:
            self._mock_index(documents)
            return

        from qdrant_client.models import PointStruct, SparseVector

        points = []
        for doc in documents:
            doc_id = doc.get("id", uuid.uuid4().hex)
            text = doc["text"]

            dense_vec = self.embedding_fn(text) if self.embedding_fn else [0.0] * EMBEDDING_DIMENSION
            sparse_data = self.sparse_fn(text) if self.sparse_fn else {"indices": [], "values": []}

            point = PointStruct(
                id=str(uuid.uuid4()),  # Qdrant은 UUID 또는 int
                vector={
                    DENSE_VECTOR_NAME: dense_vec.tolist() if hasattr(dense_vec, 'tolist') else dense_vec,
                },
                payload={
                    "doc_id": doc_id,
                    "text": text,
                    **doc.get("metadata", {}),
                },
            )
            # Sparse 벡터 별도 설정
            if sparse_data["indices"]:
                point.vector[SPARSE_VECTOR_NAME] = SparseVector(
                    indices=sparse_data["indices"],
                    values=sparse_data["values"],
                )

            points.append(point)

        self._client.upsert(
            collection_name=QDRANT_COLLECTION_NAME,
            points=points,
        )
        print(f"[QdrantHybridSearch] {len(points)}개 문서 인덱싱 완료.")

    def _mock_index(self, documents: List[Dict[str, Any]]):
        """Mock: 인메모리 저장."""
        for doc in documents:
            self._mock_documents.append({
                "id": doc.get("id", uuid.uuid4().hex[:8]),
                "text": doc["text"],
                "metadata": doc.get("metadata", {}),
            })
        print(f"[QdrantHybridSearch] Mock 인덱싱: {len(documents)}개 문서 저장.")

    # ──────────────────────────────────────────
    # 하이브리드 검색 (Dense + Sparse → RRF 병합)
    # ──────────────────────────────────────────
    def hybrid_search(
        self,
        refined_query: str,
        keywords: List[str],
        top_k: int = RERANK_TOP_K_INPUT,
    ) -> List[Dict[str, Any]]:
        """
        Dense + Sparse 동시 검색 후 RRF로 자동 병합합니다.

        Args:
            refined_query: 의미 검색용 정제된 쿼리
            keywords: BM25/키워드 검색용 키워드 리스트
            top_k: 반환할 최대 문서 수

        Returns:
            [{"id": str, "text": str, "score": float, "metadata": dict}, ...]
        """
        if self.use_mock:
            return self._mock_search(refined_query, keywords, top_k)

        return self._qdrant_hybrid_search(refined_query, keywords, top_k)

    def _qdrant_hybrid_search(
        self, refined_query: str, keywords: List[str], top_k: int
    ) -> List[Dict[str, Any]]:
        """
        client.query_points()의 Prefetch + Fusion 전략을 사용한 하이브리드 검색.
        Qdrant의 RRF(Reciprocal Rank Fusion) 기능으로 별도 병합 로직 없이 결과를 결합합니다.
        """
        from qdrant_client.models import (
            NamedVector,
            NamedSparseVector,
            SparseVector,
            Prefetch,
            FusionQuery,
            Fusion,
            QueryRequest,
            SearchRequest,
        )

        # Dense Vector 생성
        dense_vec = self.embedding_fn(refined_query)
        if hasattr(dense_vec, 'tolist'):
            dense_vec = dense_vec.tolist()

        # Sparse Vector 생성 (키워드 기반)
        keyword_text = " ".join(keywords)
        sparse_data = self.sparse_fn(keyword_text) if self.sparse_fn else {"indices": [], "values": []}

        dense_prefetch = Prefetch(
            query=dense_vec,
            using=DENSE_VECTOR_NAME,
            limit=top_k,
        )

        prefetch_list = [dense_prefetch]

        if sparse_data["indices"]:
            sparse_prefetch = Prefetch(
                query=SparseVector(
                    indices=sparse_data["indices"],
                    values=sparse_data["values"],
                ),
                using=SPARSE_VECTOR_NAME,
                limit=top_k,
            )
            prefetch_list.append(sparse_prefetch)

        res = self._client.query_points(
            collection_name=QDRANT_COLLECTION_NAME,
            prefetch=prefetch_list,
            query=FusionQuery(fusion=Fusion.RRF),
            with_payload=True,
            limit=top_k,
        )

        results = []
        for point in res.points:
            doc_id = point.payload.get("doc_id", str(point.id))
            results.append({
                "id": doc_id,
                "text": point.payload.get("text", ""),
                "score": point.score,
                "metadata": {k: v for k, v in point.payload.items() if k not in ("text", "doc_id")},
            })
        return results

    @staticmethod
    def _rrf_merge(
        batch_results: list, top_k: int, k: int = 60
    ) -> List[Dict[str, Any]]:
        """
        Reciprocal Rank Fusion으로 여러 검색 결과를 병합합니다.
        RRF Score = Σ 1 / (k + rank_i)
        """
        score_map: Dict[str, float] = {}
        doc_map: Dict[str, Dict[str, Any]] = {}

        for results in batch_results:
            for rank, hit in enumerate(results):
                doc_id = hit.payload.get("doc_id", str(hit.id))
                rrf_score = 1.0 / (k + rank + 1)

                score_map[doc_id] = score_map.get(doc_id, 0.0) + rrf_score

                if doc_id not in doc_map:
                    doc_map[doc_id] = {
                        "id": doc_id,
                        "text": hit.payload.get("text", ""),
                        "metadata": {
                            k: v for k, v in hit.payload.items()
                            if k not in ("text", "doc_id")
                        },
                    }

        # RRF 점수 내림차순 정렬
        sorted_ids = sorted(score_map, key=score_map.get, reverse=True)[:top_k]
        return [
            {**doc_map[did], "score": score_map[did]}
            for did in sorted_ids
        ]

    # ──────────────────────────────────────────
    # Mock 검색
    # ──────────────────────────────────────────
    def _mock_search(
        self, refined_query: str, keywords: List[str], top_k: int
    ) -> List[Dict[str, Any]]:
        """Mock: 키워드 매칭 기반 간이 검색."""
        results = []
        for doc in self._mock_documents:
            text = doc["text"].lower()
            query_lower = refined_query.lower()

            # 단순 키워드 매칭 점수 계산
            score = 0.0
            for kw in keywords:
                if kw.lower() in text:
                    score += 1.0
            if any(word in text for word in query_lower.split()):
                score += 0.5

            if score > 0:
                results.append({
                    "id": doc["id"],
                    "text": doc["text"],
                    "score": score,
                    "metadata": doc["metadata"],
                })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]
