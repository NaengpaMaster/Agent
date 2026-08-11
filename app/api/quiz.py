import os
from fastapi import APIRouter
from openai import OpenAI
import json

router = APIRouter(prefix="/quiz", tags=["quiz"])
client = OpenAI()
MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-.mini")

@router.get("/generate")
def generate_quiz(ingredient: str):
    prompt = f"""
    "{ingredient}"의 보관, 소비, 신선도 유지와 관련된 O/X 퀴즈를 하나 만들어줘.

    조건: 
    - 반드시 "{ingredient}"에 관한 내용이어야 해. 다른 재료를 언급하면 안 돼.
    - 일반적으로 널리 알려진 식품 안전/보관 상식만 사용해.
    - 불확실한 정보는 만들지 마.
    - 너무 뻔하지 않게, 헷갈릴 만한 내용으로 만들어줘.

    아래 JSON 형식으로만 답해:
    {{
        "statement": "O/X로 답할 수 있는 문장",
        "answer": true 또는 false,
        "explanation": "정답 이유 한 줄",
        "confidence": "high 또는 medium 또는 low"
    }}
    """

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role":"user", "content":prompt}]
    )

    result = json.loads(response.choices[0].message.content)
    return result