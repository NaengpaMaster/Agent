import base64
import json

from fastapi import HTTPException
from openai import APIError, OpenAI

from app.core.config import get_settings
from app.schemas.receipt_analysis import (
    ReceiptAnalyzeItemResponse,
    ReceiptAnalyzeRequest,
    ReceiptAnalyzeResponse,
)
from app.services.shopping_recommendation_service import _extract_usage


def analyze_receipt(request: ReceiptAnalyzeRequest) -> ReceiptAnalyzeResponse:
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
    return """
너는 영수증 OCR 엔진이다.
이미지에 실제로 보이는 텍스트만 읽어라.
이미지에 없는 상품명은 절대 추측하거나 생성하지 마라.
영수증 텍스트를 읽을 수 없거나 상품명이 보이지 않으면 rawText는 "", items는 []로 반환해라.

작업:
1. 영수증의 상품명 라인을 읽는다.
2. 식재료 또는 식품으로 볼 수 있는 상품만 items에 넣는다.
3. 브랜드명, 원산지, 용량, 중량, 가격, 바코드, 행사/할인 문구는 상품명에서 제거한다.
4. 수량이 보이면 그대로 쓰고, 수량을 알 수 없으면 "1개"로 쓴다.

반드시 아래 JSON 형식만 반환해라.
{
  "rawText": "이미지에서 실제로 읽은 전체 텍스트",
  "items": [
    {
      "name": "상품명에서 정제한 식재료명",
      "quantity": "1개"
    }
  ]
}
"""


def _to_data_url(mime_type: str, image_base64: str) -> str:
    # OpenAI Vision은 URL 또는 data URL 형식의 이미지를 받는다.
    return f"data:{mime_type};base64,{image_base64}"


def _normalize_base64(image_base64: str) -> str:
    return "".join(image_base64.split())


def _validate_base64_image(image_base64: str) -> None:
    try:
        base64.b64decode(image_base64, validate=True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="유효한 base64 이미지 문자열이 아닙니다.") from exc


def _parse_response(output_text: str, usage) -> ReceiptAnalyzeResponse:
    # 모델이 ```json ... ``` 형태로 감싸도 JSON만 꺼내 파싱한다.
    try:
        data = json.loads(_strip_code_block(output_text))
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=502, detail="OpenAI Vision 응답을 해석할 수 없습니다.") from exc

    items = [
        ReceiptAnalyzeItemResponse(
            name=item.get("name", "").strip(),
            quantity=item.get("quantity") or "1개",
        )
        for item in data.get("items", [])
        if item.get("name")
    ]

    return ReceiptAnalyzeResponse(
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


def _fallback_response() -> ReceiptAnalyzeResponse:
    # 키가 없을 때 가짜 재료를 만들면 실제 OCR처럼 오해될 수 있어 빈 결과만 반환한다.
    return ReceiptAnalyzeResponse(
        rawText="",
        items=[],
        usage=_extract_usage(None, get_settings().openai_model),
    )
