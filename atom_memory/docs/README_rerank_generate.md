# rerank_generate.py README

## 역할 (Role)
**4단계: 리랭킹 및 최종 생성**을 담당합니다. 검색 엔진에서 1차로 필터링된 문서를 리랭커(Reranker) 모델을 통해 재배열(정렬)하고, 최종 선정된 소수 정예의 문서를 바탕으로 LLM(Gemini)이 답변을 문장으로 생성(Stream 포함)합니다.

## 주요 클래스
- `Reranker`: HuggingFace Inference API를 활용 (모델: `bge-reranker-v2-m3`).
  - `rerank()`: 쿼리와 검색 결과 문서들을 인풋으로 받아 관련성 점수를 재산정하여 상위 K개를 돌려줍니다.
- `GeminiGenerator`: Google Gemini 1.5 Flash 모델 활용.
  - `generate()`: 문서 내용을 바탕으로 한 번에(Synchronous) 답변을 리턴.
  - `generate_stream()`: FastAPI 등의 비동기 프레임워크와 결합하여 스트리밍(Chunking) 형태로 클라이언트에 답변을 리턴.

## 협업 가이드 (Collaboration Guide)
1. 리랭커를 HuggingFace API가 아닌 로컬 컨테이너(vLLM 기반 등)로 교체하려면, `_hf_rerank` 메서드의 엔드포인트 URL 및 Payload 형식을 수정하세요.
2. 답변 생성에 활용되는 메인 프롬프트(답변 형식, 폰트 톤 앤 매너 등)를 변경하려면 `_build_prompt` 메서드를 수정하세요. (예: 사투리 모드, 요약 모드 추가 등)
