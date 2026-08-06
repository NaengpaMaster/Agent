# NaengpaMaster Agent

Python FastAPI 기반 AI Agent 서버입니다.

## Environment

- Python 3.12
- FastAPI
- OpenAI API

## Setup

```bash
conda create -n naengpa-agent python=3.12
conda activate naengpa-agent
pip install -r requirements.txt
cp .env.example .env
```

`.env` 파일에 실제 `OPENAI_API_KEY` 값을 입력합니다.

이미 conda 환경을 생성한 경우에는 아래 명령어부터 실행합니다.

```bash
conda activate naengpa-agent
pip install -r requirements.txt
```

## Run

```bash
uvicorn app.main:app --reload
```

## Health Check

```text
GET /health
```

## Shopping Recommendation

```text
POST /agent/v1/shopping/recommendations
```

백엔드가 냉장고 재료, 장보기 재료, 추천 후보 재료를 전달하면 Agent가 장보기 추천 목록을 반환합니다.

`OPENAI_API_KEY`가 없거나 LLM 호출에 실패하면 후보 재료 기준 fallback 응답을 반환합니다.

## Project Structure

```text
app
├── api
│   ├── health.py
│   └── shopping_recommendations.py
├── core
│   └── config.py
├── schemas
├── services
└── main.py
```
