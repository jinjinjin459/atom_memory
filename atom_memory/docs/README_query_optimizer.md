# query_optimizer.py README

## 역할 (Role)
**2단계: 쿼리 최적화**를 담당하는 부분으로, Small Language Model (SLM)을 사용하여 구어체로 들어온 질문을 검색 엔진(DB)에 유리한 형태(키워드 및 정제된 문장)로 변환해 주는 역할을 합니다.

## 주요 클래스 및 기능
- `QueryOptimizer`: Ollama 또는 vLLM과 통신하는 메인 클래스.
  - `optimize(raw_query)`: 원본 질문을 받아 `{"keywords": [...], "refined_query": "..."}` 형태의 JSON 객체로 반환.
  - `_slm_optimize()`: 실제 SLM 백엔드 API를 호출하여 프롬프트를 전송.
  - `_mock_optimize()`: 단순 정규식을 이용해 불용어를 제거하고 키워드를 추출하는 Mock 동작.

## 내부 로직 (Logic)
- 구어체 질문(예: "연차 안 쓰면 돈으로 줘?") -> 최적화 쿼리(예: "연차 미사용 수당 지급 기준") 형태로 변환.
- JSON 파싱 예외처리가 내부적으로 구현되어 있어, SLM이 규칙을 어기고 다른 형태로 응답하더라도 시스템이 죽지 않고 원본 질문으로 폴백(Fallback)합니다.

## 협업 가이드 (Collaboration Guide)
1. SLM의 프롬프트를 고도화하고 싶다면 `_slm_optimize` 메서드 안의 `prompt` 변수를 수정하세요.
2. 특정 도메인(예: 의료, 법률) 전용 불용어가 있다면 `_mock_optimize`에 등록하여 테스트 효율성을 높이세요.
