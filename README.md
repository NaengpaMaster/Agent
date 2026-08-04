# NaengpaMaster Agent

Python FastAPI 기반 AI Agent 서버입니다.

## Environment

- Python 3.12
- FastAPI
- OpenAI API

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

`.env` 파일에 실제 `OPENAI_API_KEY` 값을 입력합니다.

## Run

```bash
uvicorn app.main:app --reload
```

## Health Check

```text
GET /health
```

## Project Structure

```text
app
├── api
│   └── health.py
├── core
│   └── config.py
├── schemas
├── services
└── main.py
```
