"""
📍 4단계: 리랭킹 & 생성 (LLM)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Action 1 (Rerank): bge-reranker-v2-m3를 사용하여 검색된 상위 10개 문서 중
                    질문과 가장 밀접한 3~5개만 최종 선정.
Action 2 (Generate): 선정된 문서와 질문을 Gemini 1.5 Flash API에 전달.
Action 3 (Stream): 사용자에게 스트리밍 방식으로 최종 답변 전달.
"""

import json
from typing import List, Dict, Any, Generator

from rag_config import (
    RERANKER_MODEL_NAME,
    RERANK_TOP_K_OUTPUT,
    GEMINI_API_KEY,
    GEMINI_MODEL,
    HF_TOKEN,
)


class Reranker:
    """
    bge-reranker-v2-m3 기반 리랭커.
    HuggingFace Inference API를 사용하여 query-document 쌍의 관련성 점수를 산출합니다.
    """

    def __init__(self, use_mock: bool = True):
        self.use_mock = use_mock
        self._client = None

        if not self.use_mock:
            try:
                from huggingface_hub import InferenceClient
                self._client = InferenceClient(
                    model=RERANKER_MODEL_NAME,
                    token=HF_TOKEN,
                )
                print(f"[Reranker] HuggingFace 리랭커 연결 성공 ({RERANKER_MODEL_NAME})")
            except Exception as e:
                print(f"[Reranker] 리랭커 연결 실패: {e} — Mock 모드로 대체합니다.")
                self.use_mock = True

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = RERANK_TOP_K_OUTPUT,
    ) -> List[Dict[str, Any]]:
        """
        검색 결과에서 query와 가장 관련 높은 상위 top_k 문서를 선정합니다.

        Args:
            query: 사용자 질의
            documents: 검색 결과 리스트 [{"id", "text", "score", "metadata"}, ...]
            top_k: 최종 선정 문서 수 (3~5)

        Returns:
            리랭킹된 상위 문서 리스트
        """
        if not documents:
            return []

        if self.use_mock:
            return self._mock_rerank(query, documents, top_k)

        return self._hf_rerank(query, documents, top_k)

    def _hf_rerank(
        self, query: str, documents: List[Dict[str, Any]], top_k: int
    ) -> List[Dict[str, Any]]:
        """HuggingFace Inference API를 통한 리랭킹."""
        # query-document 쌍 생성
        pairs = [[query, doc["text"]] for doc in documents]

        try:
            # HF text-classification/reranking endpoint
            scores = self._client.post(
                json={"inputs": pairs},
                task="text-classification",
            )
            scores = json.loads(scores)

            # 각 문서에 리랭크 점수 부여
            for i, doc in enumerate(documents):
                if isinstance(scores[i], list):
                    doc["rerank_score"] = scores[i][0].get("score", 0.0)
                elif isinstance(scores[i], dict):
                    doc["rerank_score"] = scores[i].get("score", 0.0)
                else:
                    doc["rerank_score"] = float(scores[i])

        except Exception as e:
            print(f"[Reranker] HF 리랭킹 실패: {e} — 기존 점수 유지.")
            for doc in documents:
                doc["rerank_score"] = doc.get("score", 0.0)

        # 리랭크 점수 기준 내림차순 정렬 후 top_k 선정
        documents.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
        return documents[:top_k]

    def _mock_rerank(
        self, query: str, documents: List[Dict[str, Any]], top_k: int
    ) -> List[Dict[str, Any]]:
        """Mock: 기존 검색 점수를 리랭크 점수로 사용합니다."""
        for doc in documents:
            # 쿼리와 문서의 키워드 겹침 수로 간이 점수 부여
            query_terms = set(query.lower().split())
            doc_terms = set(doc["text"].lower().split())
            overlap = len(query_terms & doc_terms)
            doc["rerank_score"] = doc.get("score", 0.0) + overlap * 0.1

        documents.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
        selected = documents[:top_k]

        print(f"[Reranker] Mock 리랭킹 완료 — {len(documents)}개 중 {len(selected)}개 선정.")
        for i, doc in enumerate(selected, 1):
            print(f"  #{i} (score={doc['rerank_score']:.2f}) {doc['text'][:60]}...")
        return selected


class GeminiGenerator:
    """
    Gemini 1.5 Flash를 이용한 최종 답변 생성기.
    스트리밍 방식으로 답변을 전달합니다.
    """

    def __init__(self, use_mock: bool = True):
        self.use_mock = use_mock
        self._model = None

        if not self.use_mock:
            try:
                from google import genai
                self._client = genai.Client(api_key=GEMINI_API_KEY)
                print(f"[GeminiGenerator] Gemini API 연결 성공 ({GEMINI_MODEL})")
            except Exception as e:
                print(f"[GeminiGenerator] Gemini 연결 실패: {e} — Mock 모드로 대체합니다.")
                self.use_mock = True

    def generate(self, query: str, context_docs: List[Dict[str, Any]]) -> str:
        """동기식 답변 생성."""
        if self.use_mock:
            return self._mock_generate(query, context_docs)
        return self._gemini_generate(query, context_docs)

    def generate_stream(
        self, query: str, context_docs: List[Dict[str, Any]]
    ) -> Generator[str, None, None]:
        """
        스트리밍 방식의 답변 생성.
        FastAPI의 StreamingResponse와 결합하여 실시간 전달 가능.
        """
        if self.use_mock:
            yield self._mock_generate(query, context_docs)
            return

        yield from self._gemini_stream(query, context_docs)

    # ──────────────────────────────────────────
    # Gemini API 호출
    # ──────────────────────────────────────────
    def _build_prompt(self, query: str, context_docs: List[Dict[str, Any]]) -> str:
        context = "\n\n".join([
            f"[문서 {i+1}] (ID: {doc['id']})\n{doc['text']}"
            for i, doc in enumerate(context_docs)
        ])

        return f"""당신은 정확하고 유용한 답변을 제공하는 AI 어시스턴트입니다.
아래 제공된 문서만을 근거로 사용자의 질문에 답변하세요.

[참조 문서]
{context}

[사용자 질문]
{query}

[답변 규칙]
1. 제공된 문서에 근거한 사실만 답변하세요.
2. 문서에서 관련 정보를 찾을 수 없으면 "제공된 문서에서 해당 정보를 찾을 수 없습니다."라고 답하세요.
3. 가능한 경우 근거가 되는 문서 번호를 인용하세요.
"""

    def _gemini_generate(self, query: str, context_docs: List[Dict[str, Any]]) -> str:
        prompt = self._build_prompt(query, context_docs)
        try:
            response = self._client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
            )
            return response.text
        except Exception as e:
            return f"[Gemini 호출 에러] {e}\n\n[하지만 Qdrant 검색된 컨텍스트]\n" + "\n".join([f"- {d['text']}" for d in context_docs])

    def _gemini_stream(
        self, query: str, context_docs: List[Dict[str, Any]]
    ) -> Generator[str, None, None]:
        prompt = self._build_prompt(query, context_docs)
        response = self._client.models.generate_content_stream(
            model=GEMINI_MODEL,
            contents=prompt,
        )
        for chunk in response:
            if chunk.text:
                yield chunk.text

    # ──────────────────────────────────────────
    # Mock 생성
    # ──────────────────────────────────────────
    def _mock_generate(self, query: str, context_docs: List[Dict[str, Any]]) -> str:
        """Mock: 검색 결과를 기반으로 구조화된 답변을 생성합니다."""
        answer_parts = [f"질의: '{query}'에 대한 검색 기반 답변입니다.\n"]

        for i, doc in enumerate(context_docs, 1):
            text_preview = doc["text"][:120]
            answer_parts.append(
                f"[문서 {i} / ID: {doc['id']}]\n"
                f"  → {text_preview}...\n"
            )

        if not context_docs:
            answer_parts.append("관련 문서를 찾을 수 없습니다.")

        answer_parts.append(
            f"\n총 {len(context_docs)}개의 관련 문서를 기반으로 답변이 생성되었습니다."
        )
        return "\n".join(answer_parts)
