# rag_config.py README

## 역할 (Role)
RAG 파이프라인 전체에서 사용되는 환경변수와 상수를 중앙에서 관리하는 설정 파일입니다. 다른 개발자가 파이프라인의 모델, API 키, URL 등을 변경할 때 이 파일만 수정하면 되도록 구성되어 있습니다.

## 주요 설정 요소 (Key Configurations)
- **임베딩 모델**: `intfloat/multilingual-e5-small` (차원: 384)
- **Redis (1단계)**: Cache URL, 유사도 임계값(0.90)
- **SLM (2단계)**: Ollama/vLLM 백엔드 선택, 모델명 (`llama3:8b`)
- **Qdrant (3단계)**: Qdrant URL, API 키, Collection 이름, Dense/Sparse 벡터명
- **리랭커 (4단계)**: `BAAI/bge-reranker-v2-m3`, Top K 설정
- **최종 LLM (4단계)**: Gemini 1.5 Flash, Gemini API 키
- **HuggingFace**: Inference API용 Token

## 협업 가이드 (Collaboration Guide)
1. 로컬에서 개발할 때는 환경변수(.env 등)를 설정하여 이 파일의 값들이 오버라이드되도록 합니다.
2. 새로운 외부 API나 모델이 추가될 경우, 반드시 이 파일에 관련 상수를 먼저 정의하고 다른 모듈에서 임포트해 사용하세요 (하드코딩 금지).
