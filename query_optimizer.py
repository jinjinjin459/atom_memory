"""
📍 2단계: 쿼리 최적화 (SLM)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Setup: vLLM 또는 Ollama를 통해 Llama-3-8B 구동.
Task:  구어체 질문을 검색에 유리한 구조로 변환.
Output (JSON):
  {
    "keywords": ["연차", "미사용 수당", "지급 기준"],
    "refined_query": "회사의 연차 발생 기준 및 미사용 연차 수당 지급 규정에 대해 알려줘"
  }
"""

import json
import re
from typing import Dict, List

from rag_config import SLM_BACKEND, SLM_MODEL, SLM_BASE_URL


class QueryOptimizer:
    """
    SLM(Small Language Model)을 이용해 구어체 질의를 검색 최적화된 구조로 변환합니다.

    - Ollama 또는 vLLM 백엔드를 지원합니다.
    - SLM을 사용할 수 없을 경우 간단한 규칙 기반 Mock으로 동작합니다.
    """

    def __init__(self, use_mock: bool = True):
        self.use_mock = use_mock
        self._client = None

        if not self.use_mock:
            try:
                import requests
                # Ollama 또는 vLLM 헬스 체크
                resp = requests.get(f"{SLM_BASE_URL}/api/tags", timeout=3)
                if resp.status_code == 200:
                    self._client = "ollama"
                    print(f"[QueryOptimizer] Ollama 연결 성공 (모델: {SLM_MODEL})")
            except Exception:
                try:
                    import requests
                    resp = requests.get(f"{SLM_BASE_URL}/v1/models", timeout=3)
                    if resp.status_code == 200:
                        self._client = "vllm"
                        print(f"[QueryOptimizer] vLLM 연결 성공 (모델: {SLM_MODEL})")
                except Exception:
                    print("[QueryOptimizer] SLM 서버 연결 불가 — Mock 모드로 대체합니다.")
                    self.use_mock = True

    def optimize(self, raw_query: str) -> Dict[str, object]:
        """
        구어체 질의를 검색 최적화된 JSON으로 변환합니다.

        Returns:
            {
                "keywords": List[str],
                "refined_query": str
            }
        """
        if self.use_mock or self._client is None:
            return self._mock_optimize(raw_query)
        return self._slm_optimize(raw_query)

    # ──────────────────────────────────────────
    # SLM 호출 (Ollama / vLLM)
    # ──────────────────────────────────────────
    def _slm_optimize(self, raw_query: str) -> Dict[str, object]:
        import requests

        prompt = f"""당신은 검색 쿼리 최적화 전문가입니다.
사용자의 구어체 질문을 분석하여 아래 JSON 형식으로 변환하세요.

[규칙]
1. keywords: 검색에 핵심이 되는 명사만 추출 (3~5개)
2. refined_query: 검색에 유리하도록 구조화된 질문 (구어체 제거)

[입력 질문]
{raw_query}

[출력 형식 - 반드시 JSON만 출력]
{{"keywords": [...], "refined_query": "..."}}"""

        if self._client == "ollama":
            resp = requests.post(
                f"{SLM_BASE_URL}/api/generate",
                json={"model": SLM_MODEL, "prompt": prompt, "stream": False},
                timeout=30,
            )
            text = resp.json().get("response", "")
        else:  # vLLM (OpenAI-compatible)
            resp = requests.post(
                f"{SLM_BASE_URL}/v1/completions",
                json={"model": SLM_MODEL, "prompt": prompt, "max_tokens": 256},
                timeout=30,
            )
            text = resp.json()["choices"][0]["text"]

        return self._parse_json(text, raw_query)

    # ──────────────────────────────────────────
    # Mock 규칙 기반 최적화
    # ──────────────────────────────────────────
    def _mock_optimize(self, raw_query: str) -> Dict[str, object]:
        """SLM 없이 간단한 키워드 추출을 수행합니다."""
        # 불용어 제거
        stopwords = {"은", "는", "이", "가", "을", "를", "의", "에", "에서", "로",
                      "으로", "와", "과", "및", "대해", "대한", "해줘", "알려줘",
                      "좀", "어떻게", "무엇", "어떤", "하는", "있는", "되는",
                      "해주세요", "뭐야", "뭔가요", "요", "다", "합니다"}

        # 한글 명사/키워드 추출 (간이)
        tokens = re.findall(r'[가-힣a-zA-Z0-9]+', raw_query)
        keywords = [t for t in tokens if t not in stopwords and len(t) >= 2]

        # refined query: 불용어 제거 후 재구성
        refined = " ".join(keywords) + " 관련 정보"

        result = {
            "keywords": keywords[:5],
            "refined_query": refined,
        }
        print(f"[QueryOptimizer] Mock 최적화 결과: {json.dumps(result, ensure_ascii=False)}")
        return result

    # ──────────────────────────────────────────
    # JSON 파싱 유틸리티
    # ──────────────────────────────────────────
    @staticmethod
    def _parse_json(text: str, fallback_query: str) -> Dict[str, object]:
        """SLM 출력에서 JSON을 추출합니다."""
        # Markdown 코드블록 제거
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*', '', text)

        # JSON 부분만 추출
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group())
                if "keywords" in parsed and "refined_query" in parsed:
                    return parsed
            except json.JSONDecodeError:
                pass

        # 파싱 실패 시 폴백
        print("[QueryOptimizer] SLM JSON 파싱 실패 — 원본 질의를 사용합니다.")
        return {
            "keywords": fallback_query.split()[:5],
            "refined_query": fallback_query,
        }
