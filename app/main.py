from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

from app.api.health import router as health_router
from app.api.fridge_photo_analysis import router as fridge_photo_analysis_router
from app.api.inquiry_chat import router as inquiry_chat_router
from app.api.shopping_recommendations import router as shopping_recommendation_router
from app.api.quiz import router as quiz_router

from app.api.receipt_analysis import router as receipt_analysis_router

app = FastAPI(
    title="NaengpaMaster Agent",
    description="AI 장보기 추천 및 문의 Q&A 및 퀴즈 생성 Agent 서버",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://naengpa.com",
        "https://www.naengpa.com",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

Instrumentator().instrument(app).expose(
    app,
    endpoint="/metrics",
    include_in_schema=False,
)

app.include_router(health_router)
app.include_router(fridge_photo_analysis_router)
app.include_router(shopping_recommendation_router)
app.include_router(receipt_analysis_router)
app.include_router(inquiry_chat_router)
app.include_router(quiz_router)
