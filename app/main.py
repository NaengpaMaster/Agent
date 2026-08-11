from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.health import router as health_router
from app.api.inquiry_chat import router as inquiry_chat_router
from app.api.shopping_recommendations import router as shopping_recommendation_router

from app.api.receipt_analysis import router as receipt_analysis_router

app = FastAPI(
    title="NaengpaMaster Agent",
    description="AI 장보기 추천 및 문의 Q&A Agent 서버",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(shopping_recommendation_router)
app.include_router(receipt_analysis_router)
app.include_router(inquiry_chat_router)
