import json
import uuid
from typing import List, Dict
from models import Document, Chunk, AtomicFact

try:
    from huggingface_hub import InferenceClient
except ImportError:
    InferenceClient = None

class IngestionPipeline:
    def __init__(self, use_mock: bool = True, api_key: str = None):
        self.use_mock = use_mock
        self.api_key = api_key
        if not self.use_mock and InferenceClient is not None:
            self.client = InferenceClient(api_key=self.api_key)

    def contextual_chunking(self, doc: Document) -> List[Chunk]:
        """
        [문제 1 해결] 입력 크기 초과 대응 (Overlapping Chunking)
        [문제 2 해결] 교차 참조 연결을 위해 인접한 문맥을 포함하도록 설계
        """
        # 줄바꿈/단락 기준으로 청킹하며 이전 정보를 겹치게 설계 (연산량 증가 허용)
        paragraphs = doc.content.split('\n\n')
        chunks = []
        
        for i, text in enumerate(paragraphs):
            if not text.strip(): continue
            
            chunk_id = f"chunk_{doc.original_id}_{i}"
            
            # 이전 청크 부분 포함 (대명사 등 맥락 단절 완화)
            prev_context = paragraphs[i-1] if i > 0 else ""
            combined_text = f"[이전 맥락]: {prev_context}\n[본문]: {text}" if prev_context else text
            
            # [문제 3 해결] 필수 맥락 누락 방지를 위한 전역 메타데이터(Anchor) 주입
            metadata_header = f"문서명: {doc.filename} | 작성일: {doc.date}\n전역 요약(Anchor): {doc.summary}"
            
            chunks.append(Chunk(
                chunk_id=chunk_id,
                doc_id=doc.original_id,
                seq_index=i,
                text=combined_text.strip(),
                context_header=metadata_header,
                anchor_text=text[:50] + "..." # 앵커 텍스트
            ))
            
        return chunks

    def extract_facts(self, chunk: Chunk) -> List[AtomicFact]:
        """
        [문제 5 해결] LLM이 핵심을 잘못 해석하지 않도록 프롬프트로 명확한 규칙 제공
        """
        if self.use_mock:
            return self._mock_extract_facts(chunk)
        else:
            return self._llm_extract_facts(chunk)

    def _llm_extract_facts(self, chunk: Chunk) -> List[AtomicFact]:
        if InferenceClient is None:
            raise Exception("huggingface_hub package is not installed. Run: pip install huggingface_hub")
            
        prompt = f"""
        당신은 오차 없는 지식 추출기입니다. 연산 비용이 들더라도 가장 정확한 단위 사실(Atomic Fact)만 추출해야 합니다.

        [추출 원칙]
        1. 교차 참조: '그것', '이전 시간' 등의 대명사는 [이전 맥락]과 [전역 요약]을 참고해 명백한 고유명사로 모두 치환하세요. (문제 2 해결)
        2. 필수 맥락 보존: 출제 경향, 필수 주의사항, 시험 범위 등은 반드시 분리하여 추출하고 priority를 "high"로 지정하세요. (문제 3, 5 해결)
        3. 반환은 JSON 배열 포맷을 따르며, 각 객체는 content, timestamp(YYYY-MM-DD), priority(high/medium/low), keywords(배열)를 가져야 합니다.
        오직 JSON 형태로만 응답하고, 마크다운 코드 블록(```json ... ```) 안에 작성하세요.
        
        문맥 정보:
        {chunk.context_header}
        
        해석할 텍스트:
        {chunk.text}
        """

        response = self.client.chat_completion(
            model="google/gemma-2-9b-it",
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=800
        )
        
        try:
            content = response.choices[0].message.content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            
            result = json.loads(content.strip())
            facts_data = result.get("facts", result) 
            if isinstance(facts_data, dict):
                facts_data = [facts_data]
                
            facts = []
            for idx, item in enumerate(facts_data):
                f = AtomicFact(
                    fact_id=f"fact_{uuid.uuid4().hex[:8]}",
                    content=item.get("content", ""),
                    source_chunk_id=chunk.chunk_id,
                    source_metadata={"doc_id": chunk.doc_id, "anchor": chunk.anchor_text},
                    timestamp=item.get("timestamp", "unknown"),
                    priority=item.get("priority", "low"),
                    keywords=item.get("keywords", []),
                    logical_sequence=chunk.seq_index * 100 + idx # 청크 순서 기반으로 논리적 시간 순서(Timeline) 보장
                )
                facts.append(f)
            return facts
        except Exception as e:
            print(f"Failed to parse LLM response: {e}")
            return []

    def _mock_extract_facts(self, chunk: Chunk) -> List[AtomicFact]:
        facts = []
        text = chunk.text
        
        if "예산" in text and "100만" in text:
            facts.append(AtomicFact(
                fact_id=f"fact_{uuid.uuid4().hex[:8]}",
                content="2024년 프로젝트 초기 예산은 100만 달러로 책정되었다.",
                source_chunk_id=chunk.chunk_id,
                source_metadata={"doc_id": chunk.doc_id, "anchor": chunk.anchor_text},
                timestamp="2024-01-01",
                priority="high",
                keywords=["프로젝트", "예산", "초기"],
                logical_sequence=1
            ))
        if "예산" in text and "150만" in text:
            facts.append(AtomicFact(
                fact_id=f"fact_{uuid.uuid4().hex[:8]}",
                content="2024년 프로젝트 예산이 기존보다 증액되어 150만 달러로 확정되었다.",
                source_chunk_id=chunk.chunk_id,
                source_metadata={"doc_id": chunk.doc_id, "anchor": chunk.anchor_text},
                timestamp="2024-06-01",
                priority="high",
                keywords=["프로젝트", "예산", "증액"],
                logical_sequence=2 
            ))
        if "출제 경향" in text or "시험" in text:
            facts.append(AtomicFact(
                fact_id=f"fact_{uuid.uuid4().hex[:8]}",
                content="이번 시험의 출제 경향은 실무 응용 문제 위주입니다. (교수님 강조)",
                source_chunk_id=chunk.chunk_id,
                source_metadata={"doc_id": chunk.doc_id, "anchor": chunk.anchor_text},
                timestamp="2024-05-15",
                priority="high", # 무조건 보존되어야 할 필수 맥락
                keywords=["시험", "출제경향", "실무"],
                logical_sequence=chunk.seq_index
            ))
            
        if not facts:
            facts.append(AtomicFact(
                fact_id=f"fact_{uuid.uuid4().hex[:8]}",
                content=f"일반 정보: 해당 문서는 일반적인 프로젝트 현황을 다룹니다.",
                source_chunk_id=chunk.chunk_id,
                source_metadata={"doc_id": chunk.doc_id, "anchor": chunk.anchor_text},
                timestamp="unknown",
                priority="low",
                keywords=["기타"],
                logical_sequence=chunk.seq_index
            ))
            
        return facts
