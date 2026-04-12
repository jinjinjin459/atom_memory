"""
실제 Gemini API 호출 예시
- mock 없음, 실제 API 키로 실제 응답을 받습니다
"""

import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

print("=" * 50)
print("  실제 Gemini API 호출 데모")
print("=" * 50)
print(f"API Key: {api_key[:10]}...\n")

client = genai.Client(api_key=api_key)

# ── 예시 1: 간단한 질문 ──────────────────────────────
print("【예시 1】 한국의 수도는 어디인가요?")
response = client.models.generate_content(
    model="gemini-2.0-flash-lite",
    contents="한국의 수도는 어디인가요? 한 문장으로 답하세요."
)
print(f"  → {response.text.strip()}\n")

# ── 예시 2: 코드 생성 ────────────────────────────────
print("【예시 2】 파이썬으로 피보나치 수열 함수 작성해줘")
response2 = client.models.generate_content(
    model="gemini-2.0-flash-lite",
    contents="파이썬으로 피보나치 수열을 반환하는 짧은 함수를 작성해줘. 코드만 출력해."
)
print(f"  → {response2.text.strip()}\n")

# ── 예시 3: 임베딩 벡터 생성 ─────────────────────────
print("【예시 3】 '인공지능' 텍스트의 임베딩 벡터 생성")
emb = client.models.embed_content(
    model="text-embedding-004",
    contents="인공지능"
)
vec = emb.embeddings[0].values
print(f"  → 벡터 차원: {len(vec)}")
print(f"  → 첫 5개 값: {[round(v, 4) for v in vec[:5]]}\n")

print("=" * 50)
print("  모두 완료 — 실제 API 호출 성공!")
print("=" * 50)
