# ingestion.py / synthesis.py / models.py / demo.py README

## 개요
이 파일들은 기존 **Atomic Memory System (AMS)** 의 핵심 개념(고신뢰성 기억 및 사실 추출 기반 RAG 프로토타입)을 담은 모듈입니다. Qdrant 하이브리드 RAG 패키지와는 철학을 일부 공유하지만, 내부 구현 로직에 차이가 있습니다.

## 주요 역할
- **`models.py`**: 데이터의 틀(Schema)을 정의하는 dataclass 모음. 문서(`Document`), 잘린 조각(`Chunk`), 그리고 개별 지식(`AtomicFact`) 모델을 포함.
- **`ingestion.py`** (Phase 1): 긴 텍스트를 문맥 단절 없이 자르고(Contextual Chunking), LLM을 사용하여 개별 고유 사실(Atomic Fact)로 변환/분해하는 전처리 파이프라인.
- **`synthesis.py`** (Phase 2): 앞서 추출된 사실들을 바탕으로 타임라인/중복 제거(Clustered Resolution)를 거쳐 LLM 착각(할루시네이션)이 배제된 안전한 정답 텍스트를 병합 생성하는 파이프라인.
- **`demo.py`**: 위 AMS 사이클(Ingestion -> Synthesis)을 터미널에서 구동할 수 있는지 확인해주는 기존 검증/데모 스크립트.

## 협업 가이드
1. "고신뢰성(오차허용률 0%)" 프로젝트를 유지보수할 경우에는 이 파일들을 튜닝하세요.
2. 반면 "빠르고 유연한(Qdrant+Redis 융합) 하이브리드" 프로젝트 수정건이라면 `rag_*.py` 파일 군을 살펴보아야 합니다. 두 계층은 프로젝트 분리 관리가 가능하도록 각각 분리 설계되어 있습니다.
