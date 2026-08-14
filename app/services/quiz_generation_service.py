import os
import json
from openai import OpenAI
from app.services.quiz_generation_prompt import build_quiz_prompt

client = OpenAI()
MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")


def generate_quiz(ingredient: str) -> dict:
    prompt = build_quiz_prompt(ingredient)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}], 
        response_format={"type": "json_object"} #JSON 형식만 응답 강제
    )

    raw_text = _strip_code_block(response.choices[0].message.content)
    return json.loads(raw_text)

# LLM이 마크다운 문법(코드 블록, ```으로 감싸는 것)을 지우고 순수 JSON만 남김
def _strip_code_block(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.removeprefix("```json").removeprefix("```").strip()
        text = text.removesuffix("```").strip()
    return text