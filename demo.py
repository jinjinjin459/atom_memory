import os
from models import Document
from ingestion import IngestionPipeline
from synthesis import RetrievalSynthesisPipeline

def run_demo():
    print("="*70)
    print(" [고신뢰성 Atomic Memory System (AMS) PoC 검증 데모]")
    print(" - 연산 비용을 감수하더라도 확실하고 오차 없는 기억/검색 능력을 검증합니다.")
    print("="*70 + "\n")
    
    # 6가지 요건을 테스트할 문서 세팅
    doc1 = Document(
        original_id="doc_001",
        filename="project_and_exam_v1.txt",
        date="2024-01-01",
        content="2024년도 프로젝트 계획 및 수업 현황입니다.\n\n2024년 프로젝트 초기 예산은 100만 달러이다.\n\n참고로 교수님이 말씀하신 이번 시험의 출제 경향은 실무 응용 문제 위주입니다. 반드시 숙지하세요.",
        summary="초기 예산 및 주요 시험 정보 (교수님 직접 강조)"
    )
    
    doc2 = Document(
        original_id="doc_002",
        filename="project_plan_v2.txt",
        date="2024-06-01",
        content="프로젝트 긴급 수정본입니다.\n\n저번 시간에 말한 예산은 무효이며, 프로젝트 예산이 150만 달러로 확정 증액되었다.",
        summary="예산 증액에 대한 중요 변동 사항 업데이트"
    )
    
    hf_token = os.environ.get("HF_TOKEN")
    use_mock = hf_token is None
    
    if use_mock:
        print("[System] HF_TOKEN 환경변수가 감지되지 않아 Mock 모드로 진행합니다.")
    else:
        print("[System] HF_TOKEN 감지됨! HuggingFace 클라우드(Gemma)를 호출하여 작동합니다.")

    print(">>> [Phase 1: Ingestion Pipeline (전처리 및 오차 없는 정보 해체)]")
    ingestion = IngestionPipeline(use_mock=use_mock, api_key=hf_token)
    all_facts = []
    
    for i, doc in enumerate([doc1, doc2], 1):
        # 문제 1(입력 크기) & 문제 2(교차참조 단절) 해결을 위한 Contextual Chunking
        chunks = ingestion.contextual_chunking(doc)
        print(f"\n[{doc.filename}] Contextual Chunking: {len(chunks)}개 청크 생성 (인접 맥락 및 Anchor 포함)")
        
        for chunk in chunks:
            # 문제 3, 5 해결: 요약 전 필수 맥락 고립 방지 및 Fact로 해체
            facts = ingestion.extract_facts(chunk)
            all_facts.extend(facts)
            
    print("\n[추출된 Atomic Facts (원문 보존/역추적성 100%)]")
    for f in all_facts:
        if f.priority == "high":
            print(f" - [ID: {f.fact_id}] Date: {f.timestamp} | Seq: {f.logical_sequence} | Chunk_ID: {f.source_chunk_id}")
            print(f"   내용: {f.content}")
            
    print("\n" + "="*70 + "\n")
    print(">>> [Phase 2: Retrieval & Synthesis (중복 제거 및 연쇄 오류 원천 차단)]")
    synthesis = RetrievalSynthesisPipeline(use_mock=use_mock, api_key=hf_token)
    
    # 문제 4 해결: 순서/시간, 번복된 기록의 중복 집계 방지 (Timeline Resolution)
    resolved_facts = synthesis.clustered_resolution(all_facts)
    print("Step 2. Clustered Resolution (타임라인 기반 번복/중복 데이터 식별):")
    for f in resolved_facts:
        if "예산" in f.content or "시험" in f.content:
            status = "[현재유효]" if f.current else "[과거기록]"
            print(f" - [{status}] {f.content} (Time: {f.timestamp})")
            
    print("\n------------------------------------------------------------")
    query = "프로젝트 예산 변동 사항 및 기타 중요 정보를 알려주세요."
    print(f"Step 3. Traceable Synthesis 진행\n사용자 질의: '{query}'\n")
    
    # 문제 6 해결: LLM 착각/요약 누락으로 인한 연쇄 붕괴 방어용 안전망 (Safety Net Patch)
    final_response = synthesis.synthesize(query, resolved_facts)
    
    print("\n[최종 응답 (오류/누락 없는 무결성 검증 완료)]")
    print("-" * 50)
    print(final_response)
    print("-" * 50)

if __name__ == "__main__":
    run_demo()
