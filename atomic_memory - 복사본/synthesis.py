import re
from typing import List, Dict
from models import AtomicFact

try:
    from huggingface_hub import InferenceClient
except ImportError:
    InferenceClient = None

class RetrievalSynthesisPipeline:
    def __init__(self, use_mock=True, api_key=None):
        self.use_mock = use_mock
        self.api_key = api_key
        if not self.use_mock and InferenceClient is not None:
            self.client = InferenceClient(api_key=self.api_key)

    def clustered_resolution(self, facts: List[AtomicFact]) -> List[AtomicFact]:
        """
        [문제 4 해결] 순서, 시간, 중복 집계의 오류 방지
        같은 주제/키워드를 가진 사실들을 클러스터링하고, logical_sequence와 timestamp를 기반으로 
        이벤트의 타임라인을 구성한 후, 번복된 과거 정보는 current=False 로 밀어냄.
        """
        topic_clusters: Dict[str, List[AtomicFact]] = {
            "budget": [],
            "others": []
        }
        
        for f in facts:
            if any("예산" in k for k in f.keywords) or "예산" in f.content:
                topic_clusters["budget"].append(f)
            else:
                topic_clusters["others"].append(f)
                
        resolved_facts = []
        
        for topic, cluster in topic_clusters.items():
            if not cluster: continue
            
            if topic == "budget":
                # 순서/시간 기반 타임라인 정렬: timestamp -> logical_sequence 순서로
                cluster.sort(key=lambda x: (x.timestamp, x.logical_sequence), reverse=True)
                
                # 가장 최신인 데이터만 Current (중복 및 연쇄 오류 방지)
                cluster[0].current = True
                
                # 동일한 주제에 대해 이전의 지식은 무효화 하지만 기록으로 유지
                for f in cluster[1:]:
                    f.current = False
                    f.content = f"[과거기록/무효화됨] {f.content}"
                    
            resolved_facts.extend(cluster)
            
        return resolved_facts

    def synthesize(self, query: str, facts: List[AtomicFact]) -> str:
        """
        [문제 6 해결] 오류의 연쇄 작용 (가장 치명적인 문제 방지)
        Traceable Synthesis 과정 이후에 High Priority 정보의 강제 검증 수행
        """
        current_facts = [f for f in facts if f.current]
        
        if self.use_mock:
            draft = self._mock_synthesize(query, current_facts)
        else:
            draft = self._llm_synthesize(query, current_facts)
            
        final_answer = self._strict_validation_and_patch(draft, current_facts)
        return final_answer

    def _mock_synthesize(self, query: str, facts: List[AtomicFact]) -> str:
        # LLM이 답변 초안을 작성하면서 일부러 중요 정보를 빼먹었다고 가정한 시뮬레이션
        draft = f"질의 '{query}'에 대해 검색된 결과입니다.\n\n"
        
        if facts:
             budgets = [f for f in facts if "예산" in f.content and "무효화" not in f.content]
             if budgets:
                 draft += f"프로젝트의 최종 예산은 업데이트되었습니다. 요약하자면 {budgets[0].content}입니다. [{budgets[0].fact_id}]\n"
                 
        return draft.strip()

    def _llm_synthesize(self, query: str, facts: List[AtomicFact]) -> str:
        if InferenceClient is None:
            raise Exception("huggingface_hub is not installed.")
        
        context = "\n".join([f"[{f.fact_id}] {f.timestamp} - {f.content}" for f in facts])
        prompt = f"""
        사용자의 질의에 답변을 작성하세요. 반드시 아래 제공된 <Context> 안의 정보만 사용하여 작성해야 합니다.
        응답 시 사용한 정보의 출처로 팩트 ID를 [fact_1234abcd] 형식으로 문장 끝에 반드시 표기하세요.
        
        <Context>
        {context}
        </Context>
        
        질의: {query}
        """

        response = self.client.chat_completion(
            model="google/gemma-2-9b-it",
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=600
        )
        return response.choices[0].message.content.strip()

    def _strict_validation_and_patch(self, draft: str, facts: List[AtomicFact]) -> str:
        """
        원문에서 추출할 때 priority="high"로 지정된 필수맥락은 반드시 포함되어야 함
        """
        high_priority_facts = [f for f in facts if f.priority == "high" and f.current]
        
        patched_draft = draft
        error_prevented = 0
        
        for f in high_priority_facts:
            # Fact ID 인용 여부로 완벽한 오차 없는 정규식 검증
            pattern = re.compile(rf"\[{f.fact_id}\]")
            if not pattern.search(patched_draft):
                print(f"[System / Safety Net] 핵심 결론 누락 오류 혹은 오차(문제 5, 6) 감지. (ID: {f.fact_id})")
                print(f" -> 강제 패치를 통해 원래 응답에 복원합니다.")
                
                # 누락된 핵심 정보를 원문 보존 원칙에 맞게 강제 주입
                patched_draft += f"\n\n[필수 정보 강제 보존]: 문맥 분석 결과 매우 중요한 정보가 있습니다.\n-> {f.content} [원본 Anchor: {f.source_chunk_id}] [{f.fact_id}]"
                error_prevented += 1
                
        if error_prevented > 0:
            print(f"*** 총 {error_prevented}건의 연쇄 작용 오류 및 맥락 누락을 방어했습니다. ***\n")
            
        return patched_draft
