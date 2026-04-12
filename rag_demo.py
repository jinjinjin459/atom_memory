"""
[Demo] Qdrant 기반 하이브리드 RAG 데모 스크립트
================================================

[실행 방법]
  python rag_demo.py            # mock 모드 (기본값 -- API 키 불필요)
  python rag_demo.py --real     # 실제 API 모드 (.env 파일 필요)

[실제 API 모드 준비]
  1. .env.example 을 복사해 .env 를 만든다
  2. GEMINI_API_KEY, HF_TOKEN, QDRANT_API_KEY, QDRANT_URL 를 채운다
  3. python rag_demo.py --real

파이프라인 4단계:
  [1] 시맨틱 캐시   -> Redis Stack  (캐시 히트 시 즉시 반환)
  [2] 쿼리 정제     -> SLM (Llama-3-8B / Ollama)
  [3] 하이브리드 검색 -> Qdrant (Dense + Sparse + RRF)
  [4] 리랭크 + 생성  -> bge-reranker + Gemini Flash
"""

import sys
import time
import random

# ──────────────────────────────────────────────────────────────
# 실 모드 flag
# ──────────────────────────────────────────────────────────────
REAL_MODE = "--real" in sys.argv


# ──────────────────────────────────────────────────────────────
# 실 모드에서만 임포트
# ──────────────────────────────────────────────────────────────
if REAL_MODE:
    import os
    from dotenv import load_dotenv
    from google import genai
    from rag_config import GEMINI_API_KEY, EMBEDDING_MODEL_NAME, EMBEDDING_DIMENSION
    from rag_pipeline import HybridRAGPipeline

    load_dotenv()

    if not GEMINI_API_KEY:
        print("❌ GEMINI_API_KEY 가 설정되지 않았습니다. .env 파일을 확인하세요.")
        sys.exit(1)

    _gemini_client = genai.Client(api_key=GEMINI_API_KEY)

    def real_embedding_fn(text: str):
        """실제 Gemini Embedding API 호출"""
        if not text:
            return [0.0] * EMBEDDING_DIMENSION
        try:
            res = _gemini_client.models.embed_content(
                model=EMBEDDING_MODEL_NAME,
                contents=text,
            )
            return res.embeddings[0].values
        except Exception as e:
            print(f"  ⚠️  Embedding 오류: {e}")
            return [0.0] * EMBEDDING_DIMENSION

    def mock_sparse_fn(text: str):
        """Sparse 벡터 — BM25 근사 (해시 기반)"""
        idx_map = {}
        for w in text.split():
            idx = abs(hash(w)) % 10_000
            idx_map[idx] = idx_map.get(idx, 0.0) + 1.0
        return {"indices": list(idx_map.keys()), "values": list(idx_map.values())}


# ──────────────────────────────────────────────────────────────
# Mock 응답 데이터베이스 (mock 모드용)
# ──────────────────────────────────────────────────────────────
MOCK_ANSWERS = {
    "연차": (
        "[연차 수당 정책]\n"
        "  - 1년 근속 시 연차 15일 발생 (입사일 기준)\n"
        "  - 미사용 연차는 연말 정산 시 통상임금 기준으로 수당 지급\n"
        "  - 퇴직 시 잔여 연차도 정산됩니다.",
        "rag"
    ),
    "예산": (
        "[2024년 프로젝트 예산 변동]\n"
        "  - 초기 배정: 100만 달러\n"
        "  - 상반기 실적 평가 후 -> 150만 달러로 증액 확정\n"
        "  - 추가분은 R&D 및 인프라 구축에 배분",
        "rag"
    ),
    "시험": (
        "[기말시험 정보]\n"
        "  - 출제 범위: 3장 ~ 7장\n"
        "  - 출제 경향: 실무 응용 문제 위주\n"
        "  - 서술형 비중: 60% (높음 -- 꼭 대비하세요!)",
        "rag"
    ),
    "오메가": (
        "[프로젝트 오메가]\n"
        "  - 목표: 범용 인공지능(AGI) 핵심 모듈 개발\n"
        "  - 완수 기한: 2026년 12월\n"
        "  - 배정 예산: 1,000억 원",
        "rag"
    ),
    "캐시": (
        "[캐시 히트] 이미 처리된 동일 질의입니다.\n"
        "  -> 캐시에서 즉시 반환 (LLM 호출 없음, 비용 0원)",
        "cache"
    ),
}

_cache_store: dict = {}  # mock 캐시 저장소


def mock_query(question: str) -> dict:
    """API 없이 로컬에서 동작하는 mock RAG 파이프라인"""
    start = time.time()
    time.sleep(random.uniform(0.05, 0.15))  # 네트워크 지연 시뮬레이션

    # 캐시 확인
    if question in _cache_store:
        elapsed = (time.time() - start) * 1000
        answer, _ = MOCK_ANSWERS["캐시"]
        return {"answer": answer, "source": "cache", "latency_ms": elapsed}

    # 키워드 매칭으로 답변 선택
    answer_text, source = "❓ 관련 문서를 찾지 못했습니다.", "rag"
    for keyword, (ans, src) in MOCK_ANSWERS.items():
        if keyword != "캐시" and keyword in question:
            answer_text, source = ans, src
            break

    _cache_store[question] = True
    elapsed = (time.time() - start) * 1000
    return {"answer": answer_text, "source": source, "latency_ms": elapsed}


# ──────────────────────────────────────────────────────────────
# 샘플 문서
# ──────────────────────────────────────────────────────────────
SAMPLE_DOCUMENTS = [
    {
        "id": "doc_hr_001",
        "text": (
            "회사의 연차 휴가는 입사일 기준으로 1년 근속 시 15일이 발생합니다. "
            "미사용 연차에 대해서는 연말 정산 시 수당으로 지급됩니다. "
            "지급 기준은 통상임금 기반이며, 퇴직 시에도 미사용 연차 수당이 정산됩니다."
        ),
        "metadata": {"department": "HR", "category": "연차", "updated": "2024-03-01"},
    },
    {
        "id": "doc_hr_002",
        "text": (
            "육아휴직은 만 8세 이하 자녀가 있는 직원에게 최대 1년간 부여됩니다. "
            "급여는 통상임금의 80%가 고용보험에서 지급되며, "
            "복직 후 동일 직급으로 복귀하는 것이 보장됩니다."
        ),
        "metadata": {"department": "HR", "category": "육아휴직", "updated": "2024-01-15"},
    },
    {
        "id": "doc_it_001",
        "text": (
            "사내 VPN 접속 시 OTP 인증이 필수이며, 2024년 4월부터 "
            "기존 SMS 인증은 폐지되고 앱 기반 TOTP로 전환됩니다. "
            "VPN 연결 후 내부망 접속까지 평균 3초 이내입니다."
        ),
        "metadata": {"department": "IT", "category": "VPN", "updated": "2024-04-01"},
    },
    {
        "id": "doc_finance_001",
        "text": (
            "2024년 프로젝트 초기 예산은 100만 달러로 책정되었습니다. "
            "상반기 실적 평가 후 프로젝트 예산은 150만 달러로 증액 확정되었습니다. "
            "추가 예산은 R&D와 인프라 구축에 배분됩니다."
        ),
        "metadata": {"department": "Finance", "category": "예산", "updated": "2024-06-15"},
    },
    {
        "id": "doc_exam_001",
        "text": (
            "이번 학기 기말시험의 출제 경향은 실무 응용 문제 위주입니다. "
            "교수님이 직접 강조하신 시험 범위는 3장~7장이며, "
            "서술형 비중이 60%로 높아졌습니다. 반드시 숙지하세요."
        ),
        "metadata": {"department": "Education", "category": "시험", "updated": "2024-05-15"},
    },
    {
        "id": "doc_secret_001",
        "text": (
            "비밀 프로젝트인 '프로젝트 오메가'는 2026년 12월까지 완수되어야 하며, "
            "최종 목표는 범용 인공지능(AGI)의 핵심 모듈을 개발하는 것이다. "
            "예산은 1000억원이 할당되었다."
        ),
        "metadata": {"department": "AI_Research", "category": "Secret", "updated": "2026-04-12"},
    },
]


# ──────────────────────────────────────────────────────────────
# 데모 실행
# ──────────────────────────────────────────────────────────────
def run_demo():
    mode_label = "[실제 API]" if REAL_MODE else "[Mock]"
    print("=" * 70)
    print(f"  Qdrant 기반 하이브리드 RAG 데모  {mode_label}")
    print("=" * 70)

    if REAL_MODE:
        pipeline = HybridRAGPipeline(
            use_mock=False,
            embedding_fn=real_embedding_fn,
            sparse_fn=mock_sparse_fn,
        )
        pipeline.index_documents(SAMPLE_DOCUMENTS)
        query_fn = pipeline.query
    else:
        print("\n  [INFO] API 키 없이 로컬에서 실행 중입니다.")
        print("         실제 API 연동: python rag_demo.py --real\n")
        query_fn = mock_query

    # ── 테스트 질의 목록 ──────────────────────────────────────
    tests = [
        ("연차 관련 질의",           "우리 회사에서 연차 안 쓰면 수당으로 받을 수 있어?"),
        ("동일 질의 (캐시 확인)",    "우리 회사에서 연차 안 쓰면 수당으로 받을 수 있어?"),
        ("프로젝트 예산 질의",       "프로젝트 현재 예산이 얼마야? 변동 사항 알려줘"),
        ("기말시험 범위 질의",       "시험 범위가 어디까지야? 출제 경향도 알려줘"),
        ("프로젝트 오메가 질의",     "프로젝트 오메가의 목표는 무엇이고 예산은 얼마나 되나요?"),
    ]

    for i, (title, question) in enumerate(tests, 1):
        print(f"\n{'=' * 70}")
        print(f"  [Q{i}] {title}")
        print(f"  질문: {question}")
        print(f"  {'─' * 66}")

        result = query_fn(question)

        print(f"\n  [답변]\n{result['answer']}\n")
        hit = "(CACHE HIT)" if result["source"] == "cache" else ""
        print(f"  소스: {result['source']} {hit}  |  응답시간: {result['latency_ms']:.1f}ms")
        if result["source"] == "cache":
            print("  >> 캐시 히트 -- LLM 호출 없음, 비용 절감!")

    # ── 인프라 요약 ───────────────────────────────────────────
    print(f"\n\n{'=' * 70}")
    print("  [인프라 구성 요약]")
    print(f"  {'─' * 66}")
    rows = [
        ("캐시 레이어",  "Redis Stack",                  "시맨틱 유사도 기반 캐싱 (임계값 0.90)"),
        ("쿼리 정제",   "SLM - Llama-3-8B (Ollama)",     "불명확한 질의 재기술"),
        ("벡터 검색",   "Qdrant Cloud - Hybrid (RRF)",    "Dense + Sparse 융합 검색"),
        ("리랭크",      "bge-reranker-v2-m3 (HF)",       "Top-10 -> Top-5 정밀 필터링"),
        ("응답 생성",   "Gemini Flash (Google AI)",       "저비용 고성능 최종 생성"),
    ]
    print(f"  {'계층':<12}{'기술':<35}{'역할'}")
    print(f"  {'─' * 66}")
    for layer, tech, role in rows:
        print(f"  {layer:<12}{tech:<35}{role}")
    print(f"  {'─' * 66}\n")


if __name__ == "__main__":
    run_demo()
