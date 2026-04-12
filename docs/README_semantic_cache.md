# semantic_cache.py README

## 역할 (Role)
**1단계: 시맨틱 캐싱**을 담당하는 모듈입니다. 사용자의 질문을 임베딩 벡터로 변환한 뒤, Redis Stack을 사용하여 기존에 유사한 질문(유사도 0.90 이상)이 있었는지 초고속으로 검색합니다.

## 주요 클래스 및 함수
- `SemanticCache`: Redis 연결 및 벡터 검색을 관리하는 메인 클래스.
  - `_ensure_index()`: Redis에 벡터 검색용 인덱스를 동적으로 생성.
  - `lookup(query)`: 질문의 캐시 적중 여부를 확인하고, 적중 시 답변을 즉시 반환.
  - `store(query, answer)`: 새로 생성된 질문과 답변을 캐시에 저장.

## 특징 (Features)
- **자동 폴백 (Fallback)**: Redis 서버에 연결할 수 없거나 장애가 발생한 경우, 에러를 발생시키지 않고 메모리 기반의 Mock 객체로 동작(Fallback)하도록 설계되어 로컬 테스트가 매우 편리합니다.
- **초고속 응답**: LLM을 거치지 않으므로 비용과 지연시간을 획기적으로 줄입니다.

## 협업 가이드 (Collaboration Guide)
1. Redis의 검색 인덱스 구조를 변경해야 한다면 `_ensure_index` 및 `_redis_lookup` 메서드를 수정하세요.
2. 현재 `Cosine Similarity`를 기준으로 0.90 이상의 임계값을 가집니다. 프로젝트 특성에 따라 `rag_config.py`의 `CACHE_SIMILARITY_THRESHOLD`를 조절하세요.
