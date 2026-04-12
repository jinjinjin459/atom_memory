# 고신뢰성 Atomic Memory System (AMS) PoC

에너지(연산 비용)를 더 소모하더라도, 대량의 데이터에서 오차 없이 정확한 정보만을 관리/추출/응답하는 것을 목표로 설계된 기억 검색 및 합성 시스템입니다.

## 6가지 핵심 문제점 해결 아키텍처

1. **입력 크기 초과 대응 (Contextual Chunking)**
   - 파일이 LLM의 처리 한계를 넘어설 때 발생하는 단절을 막기 위해, 슬라이딩 윈도우 방식으로 주변 맥락 조각을 지닌 `Chunk`를 구성합니다.

2. **교차 참조 및 맥락 연결의 단절 극복**
   - 분리된 파일 속의 '그것', '저번 시간' 등 지시대명사를 문맥 데이터(Context Header/Global Anchor)를 참조해 원칙적으로 고유명사로 완전해석 후 Fact 객체로 저장합니다.

3. **요약 과정에서의 필수 맥락 누락 방지 (Anchor 원칙)**
   - 요약본에 의존하여 필수 조건(시험범위 등)이 휘발되는 것을 막기 위해, 원문에서 즉각 독립 사실을 도출하되, 원래의 `source_chunk_id`를 끝까지 보존합니다.

4. **순서, 시간, 중복 집계의 오류 제어 (Clustered Resolution)**
   - 여러 곳에 흩어져있거나 번복된 사건들의 논리적 순서(`logical_sequence`)와 시간(`timestamp`)을 추적합니다. 번복 처리된 내용은 `Current=False`로 무효화되어 최종 집계에서 중복/오류를 방지합니다.

5. **핵심 결론 파악 오류 차단 (Traceable Priority)**
   - 중요한 출제경향, 결론, 제약사항은 추출 시점부터 `priority=high`로 관리됩니다. 단순히 LLM에게 요약을 전적으로 맡기지 않습니다.

6. **오류의 연쇄 작용 방어 (Safety Net Validation)**
   - LLM이 답변 생성 중 핵심 정보를 빼먹거나, 지시대명사를 착각했더라도 응답 마지막 계층(Validation 단계)에서 `[Fact_ID]`의 등장을 엄격한 Regex로 검사하고, 유효한 핵심 정보가 누락되었다면 답변 말미에 강제로 원문 내용을 복원/보존(Patch)합니다.

---

## 🛠️ Qdrant 기반 단순화된 하이브리드 RAG 구현 명세

### 📍 1단계: API 게이트웨이 & 시맨틱 캐싱

| 항목 | 설명 |
|------|------|
| **Logic** | 모든 질의는 FastAPI를 거쳐 Redis에서 1차 필터링 |
| **임베딩** | 사용자 질문을 `e5-small-v2` 모델로 임베딩 |
| **캐시 조회** | Redis Stack의 Vector Search로 유사도 **0.90 이상** 과거 질문 확인 |
| **적중 시** | 즉시 답변 반환 (LLM 비용 0원, 지연시간 0.1초 미만) |

> 📂 구현 파일: `semantic_cache.py`, `rag_api.py`

### 📍 2단계: 쿼리 최적화 (SLM)

| 항목 | 설명 |
|------|------|
| **Setup** | vLLM 또는 Ollama를 통해 Llama-3-8B 구동 |
| **Task** | 구어체 질문을 검색에 유리한 구조로 변환 |

**출력 형식 (JSON):**
```json
{
  "keywords": ["연차", "미사용 수당", "지급 기준"],
  "refined_query": "회사의 연차 발생 기준 및 미사용 연차 수당 지급 규정에 대해 알려줘"
}
```

> 📂 구현 파일: `query_optimizer.py`

### 📍 3단계: Qdrant 통합 하이브리드 검색 (Single DB)

| 항목 | 설명 |
|------|------|
| **Tech** | Qdrant (Cloud 또는 Docker) |
| **Key Setup** | 단일 Collection 내 Named Vectors 설정 (Dense + Sparse) |
| **Dense** | `refined_query` 기반 의미 검색 |
| **Sparse** | `keywords` 기반 키워드/BM25 검색 |
| **검색** | `client.search_batch()`로 두 가지 검색 동시 수행 |
| **병합** | Qdrant의 **RRF(Reciprocal Rank Fusion)** 기능으로 자동 병합 |

> 📂 구현 파일: `qdrant_search.py`

### 📍 4단계: 리랭킹 & 생성 (LLM)

| 단계 | 설명 |
|------|------|
| **Rerank** | `bge-reranker-v2-m3`로 상위 10개 중 3~5개 최종 선정 |
| **Generate** | 선정 문서 + 질문을 **Gemini 1.5 Flash** API에 전달 |
| **Stream** | 스트리밍 방식으로 최종 답변 전달 |

> 📂 구현 파일: `rerank_generate.py`

### 📊 인프라 요약 테이블 (Qdrant 중심)

| 계층 | 사용 기술 | 비고 |
|------|-----------|------|
| **Cache** | Redis Stack + e5-small | 초고속 캐싱 및 단순 벡터 검색 |
| **Logic/SLM** | FastAPI + vLLM (Llama-3-8B) | 쿼리 정제 및 핵심 로직 제어 |
| **Storage** | Qdrant Cloud | API 기반 운영 가능, 하이브리드 검색 일원화 |
| **Reranker** | HuggingFace Inference | 최종 검색 결과 정제 (정확도 핵심) |
| **Final LLM** | Gemini 1.5 Flash | 저비용/고성능 생성 API |

---

## 프로젝트 구조

```
asdfasdf-main - 복사본/
├── models.py           # 데이터 모델 (Document, Chunk, AtomicFact)
├── ingestion.py        # AMS Phase 1: 전처리 파이프라인
├── synthesis.py        # AMS Phase 2: 검색 & 합성 파이프라인
├── demo.py             # AMS 데모 스크립트
│
├── rag_config.py       # [RAG] 환경변수 & 상수 설정
├── semantic_cache.py   # [RAG] 1단계: Redis 시맨틱 캐시
├── query_optimizer.py  # [RAG] 2단계: SLM 쿼리 최적화
├── qdrant_search.py    # [RAG] 3단계: Qdrant 하이브리드 검색
├── rerank_generate.py  # [RAG] 4단계: 리랭킹 & Gemini 생성
├── rag_pipeline.py     # [RAG] 전체 파이프라인 오케스트레이터
├── rag_api.py          # [RAG] FastAPI API 게이트웨이
├── rag_demo.py         # [RAG] 하이브리드 RAG 데모 스크립트
│
├── requirements.txt    # 의존성 패키지 목록
└── README.md           # 프로젝트 문서
```

## 실행 방법

### AMS 데모
```bash
python demo.py
```

### Qdrant 하이브리드 RAG 데모 (Mock 모드)
```bash
python rag_demo.py
```

### FastAPI 서버 실행
```bash
uvicorn rag_api:app --reload --port 8000
```

### 환경변수 설정 (프로덕션)
```bash
# Redis
export REDIS_URL="redis://localhost:6379"

# SLM (Ollama 또는 vLLM)
export SLM_BACKEND="ollama"
export SLM_MODEL="llama3:8b"
export SLM_BASE_URL="http://localhost:11434"

# Qdrant
export QDRANT_URL="http://localhost:6333"
export QDRANT_API_KEY="your-qdrant-api-key"
export QDRANT_COLLECTION="hybrid_rag_docs"

# Gemini
export GEMINI_API_KEY="your-gemini-api-key"

# HuggingFace (리랭커)
export HF_TOKEN="your-hf-token"
```
