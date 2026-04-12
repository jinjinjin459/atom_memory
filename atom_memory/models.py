from dataclasses import dataclass, field
from typing import Dict, Any, List

@dataclass
class Document:
    original_id: str
    filename: str
    date: str
    content: str
    summary: str # 앵커 역할 (원문 검색용)

@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    seq_index: int
    text: str
    context_header: str # 전역 맥락 유지
    anchor_text: str    # 청크 자체 요약 (교차 참조용)

@dataclass
class AtomicFact:
    fact_id: str
    content: str
    source_chunk_id: str  # 1. 원문 보존 원칙 (항상 원문으로 역추적 가능)
    source_metadata: Dict[str, Any]
    timestamp: str        # 4. 시간/순서 오류 방지용 타임스탬프
    priority: str         # 5. 핵심 결론 오류 방지 (필수 정보는 high)
    keywords: List[str] = field(default_factory=list) # 중복 집계 방지용 클러스터링 키워드
    current: bool = True
    logical_sequence: int = 0 # 사건의 연속성 보장을 위한 논리적 순서

    def to_dict(self):
        return {
            "fact_id": self.fact_id,
            "content": self.content,
            "source_chunk_id": self.source_chunk_id,
            "source_metadata": self.source_metadata,
            "timestamp": self.timestamp,
            "priority": self.priority,
            "keywords": self.keywords,
            "current": self.current,
            "logical_sequence": self.logical_sequence
        }
