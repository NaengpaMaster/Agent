from fastapi import APIRouter
from app.services.quiz_generation_service import generate_quiz

router = APIRouter(prefix="/agent/v1/quiz", tags=["quiz"])


@router.get("/generate")
def generate_quiz_endpoint(ingredient: str):
    return generate_quiz(ingredient)