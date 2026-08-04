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
