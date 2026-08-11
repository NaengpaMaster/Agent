from fastapi import FastAPI

from app.api.health import router as health_router
from app.api.inquiry_chat import router as inquiry_chat_router
from app.api.shopping_recommendations import router as shopping_recommendation_router
from app.api.quiz import router as quiz_router

from app.api.receipt_analysis import router as receipt_analysis_router

app = FastAPI(
    title="NaengpaMaster Agent",
    description="AI 장보기 추천 및 문의 Q&A 및 퀴즈 생성 Agent 서버",
    version="0.1.0",
)

app.include_router(health_router)
app.include_router(shopping_recommendation_router)
app.include_router(receipt_analysis_router)
app.include_router(inquiry_chat_router)
app.include_router(quiz_router)