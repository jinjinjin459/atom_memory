# qdrant_search.py README

## 역할 (Role)
**3단계: 하이브리드 검색**을 담당합니다. Vector DB인 Qdrant를 활용하여, Dense 검색(의미론적 검색)과 Sparse 검색(키워드 기반 BM25 검색)을 단일 컬렉션에서 동시에 수행합니다.

## 주요 클래스 및 기능
- `QdrantHybridSearch`: Qdrant 클라이언트와의 세션을 관리합니다.
  - `_ensure_collection()`: `Named Vectors` (Dense 및 Sparse) 설정을 기반으로 Qdrant 컬렉션을 초기화합니다.
  - `index_documents()`: 대량의 문서를 DB에 삽입(Upsert)합니다.
  - `hybrid_search()`: `search_batch`를 사용해 의미 검색과 키워드 검색을 동시에 호출한 뒤, RRF(Reciprocal Rank Fusion) 알고리즘으로 결과 순위를 동적으로 결합(Merge)합니다.

## 특징 (Features)
- **Reciprocal Rank Fusion (RRF)**: 서로 다른 두 검색 방식의 결과를 자동으로 병합하여 검색 정확도를 극대화합니다. 추가적인 점수 병합 로직을 수동으로 짤 필요가 없습니다.
- **Mock 폴백**: Qdrant Cloud나 로컬 컨테이너 연결 실패 시, 인메모리 리스트 검색 모드로 자동 전환됩니다.

## 협업 가이드 (Collaboration Guide)
1. 새로운 메타데이터 필터링(예: 부서별 검색, 날짜 범위 검색)을 추가하려면, `_qdrant_hybrid_search`의 `SearchRequest` 내부의 `filter` 파라미터를 구현하세요.
2. 사용하는 Sparse 벡터 알고리즘이 변경된다면 인덱싱(`index_documents`) 및 검색 입력부 구조를 업데이트해야 합니다.
