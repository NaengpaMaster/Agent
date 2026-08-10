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

    return _parse_response(response.output_text)


def _build_prompt() -> str:
    return """
영수증 이미지에서 식재료 후보만 추출해.
브랜드명, 상품 시리즈명, 원산지, 바코드, 제조사명, 용량, 중량, 가격, 행사 문구, 할인 문구는 제거해.
가능하면 순수 식재료명만 반환해.
예: 국산 팽이버섯 -> 팽이버섯
예: 찹쌀이 콩나물 -> 콩나물
예: 빙그레 굿모닝우유 저지방 -> 우유
상품명이 식재료가 아니면 제외해.
수량이 보이면 그대로 쓰고, 수량을 알 수 없으면 "1개"로 써.
응답은 설명 없이 JSON 객체만 반환해.

형식:
{
  "rawText": "영수증에서 읽은 전체 텍스트",
  "items": [
    {
      "name": "감자",
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


def _parse_response(output_text: str) -> ReceiptAnalyzeResponse:
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
    )


def _strip_code_block(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.removeprefix("```json").removeprefix("```").strip()
        text = text.removesuffix("```").strip()
    return text


def _fallback_response() -> ReceiptAnalyzeResponse:
    # 로컬 연결 확인용. OPENAI_API_KEY가 있으면 실제 Vision 호출 경로를 탄다.
    return ReceiptAnalyzeResponse(
        rawText="fallback OCR result",
        items=[
            ReceiptAnalyzeItemResponse(
                name="감자",
                quantity="1개",
            )
        ],
    )
