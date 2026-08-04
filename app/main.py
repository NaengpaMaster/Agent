from fastapi import FastAPI

from app.api.health import router as health_router

app = FastAPI(
    title="NaengpaMaster Agent",
    description="AI 장보기 추천 Agent 서버",
    version="0.1.0",
)

app.include_router(health_router)
