import base64
import json

from fastapi import HTTPException
from openai import APIError, OpenAI

from app.core.config import get_settings
from app.schemas.fridge_photo_analysis import (
    FridgePhotoAnalyzeItemResponse,
    FridgePhotoAnalyzeRequest,
    FridgePhotoAnalyzeResponse,
)
from app.services.ingredient_vision_prompt import build_ingredient_vision_prompt
from app.services.shopping_recommendation_service import _extract_usage


def analyze_fridge_photo(request: FridgePhotoAnalyzeRequest) -> FridgePhotoAnalyzeResponse:
    settings = get_settings()
    if not settings.openai_api_key:
        return _fallback_response()

    image_base64 = _normalize_base64(request.image_base64)
    _validate_base64_image(image_base64)

    client = OpenAI(api_key=settings.openai_api_key)
    try:
        response = client.responses.create(
            model=settings.openai_model,
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": _build_prompt(),
                        },
                        {
                            "type": "input_image",
                            "image_url": _to_data_url(request.mime_type, image_base64),
                        },
                    ],
                }
            ],
        )
    except APIError as exc:
        raise HTTPException(status_code=502, detail="OpenAI Vision 호출에 실패했습니다.") from exc

    return _parse_response(response.output_text, _extract_usage(response, settings.openai_model))


def _build_prompt() -> str:
    return build_ingredient_vision_prompt("""
냉장고 내부 사진 또는 식재료 사진에서 냉장고에 등록할 수 있는 식재료 후보만 추출해.
브랜드명, 포장 문구, 원산지, 용량, 중량, 가격, 행사 문구는 제거해.
포장된 가공식품은 포장지에 적힌 대표 상품군으로 판단해. 글자가 보이는 상품명은 시각적 일부 모양보다 상품명을 우선해.
같은 재료가 여러 개 보이면 하나로 합치고 수량을 추정해.
불확실한 물체, 조리도구, 용기, 사람이 먹는 식재료가 아닌 것은 제외해.
예: 신라면 봉지 여러 개 -> 라면
예: 짜파게티 봉지 -> 라면
예: 서울우유 -> 우유
""")


def _to_data_url(mime_type: str, image_base64: str) -> str:
    return f"data:{mime_type};base64,{image_base64}"


def _normalize_base64(image_base64: str) -> str:
    return "".join(image_base64.split())


def _validate_base64_image(image_base64: str) -> None:
    try:
        base64.b64decode(image_base64, validate=True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="유효한 base64 이미지 문자열이 아닙니다.") from exc


def _parse_response(output_text: str, usage) -> FridgePhotoAnalyzeResponse:
    try:
        data = json.loads(_strip_code_block(output_text))
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=502, detail="OpenAI Vision 응답을 해석할 수 없습니다.") from exc

    items = [
        FridgePhotoAnalyzeItemResponse(
            name=item.get("name", "").strip(),
            quantity=item.get("quantity") or "1개",
        )
        for item in data.get("items", [])
        if item.get("name")
    ]

    return FridgePhotoAnalyzeResponse(
        rawText=data.get("rawText", ""),
        items=items,
        usage=usage,
    )


def _strip_code_block(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.removeprefix("```json").removeprefix("```").strip()
        text = text.removesuffix("```").strip()
    return text


def _fallback_response() -> FridgePhotoAnalyzeResponse:
    return FridgePhotoAnalyzeResponse(
        rawText="fallback fridge photo result",
        items=[
            FridgePhotoAnalyzeItemResponse(
                name="감자",
                quantity="1개",
            )
        ],
        usage=_extract_usage(None, get_settings().openai_model),
    )
