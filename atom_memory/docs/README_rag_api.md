# rag_api.py README

## 역할 (Role)
FastAPI를 활용해 RAG 파이프라인에 웹으로 접근할 수 있도록 REST API 엔드포인트를 열어주는 **API 게이트웨이 / 프론트엔드 통신 창구** 모듈입니다.

## 제공되는 엔드포인트
- `GET /health` : 서버 작동 및 파이프라인 초기화 상태 확인 (로드밸런서 헬스체크용)
- `POST /query` : 사용자 질의를 전송하고 최종 RAG 응답 데이터(대기 시간 포함)를 동기식으로 반환받습니다.
- `POST /query/stream`: 스트리밍 방식(`StreamingResponse`)으로 텍스트를 실시간으로 조각조각 반환받아, 타이핑 치는 듯한 UI 연출이 가능합니다.
- `POST /index` : 초기 문서 더미들을 Qdrant Storage에 밀어넣을(Push) 때 사용합니다.

## 협업 가이드 (Collaboration Guide)
1. 프론트엔드 개발자와 API 스펙을 논의할 때 이 소스 내의 `QueryRequest`, `QueryResponse` Pydantic 모델을 공유하세요.
2. JWT, CORS 설정 혹은 로깅(Middleware) 계층 추가가 필요하다면 이 파일(`app = FastAPI(...)` 선언부 인근)에 작업하세요.
