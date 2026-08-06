import json
from decimal import Decimal

from openai import OpenAI

from app.core.config import get_settings
from app.schemas.shopping_recommendation import (
    AgentProduct,
    LlmUsageResponse,
    ShoppingRecommendationItemResponse,
    ShoppingRecommendationRequest,
    ShoppingRecommendationResponse,
)


def recommend_shopping_items(
    request: ShoppingRecommendationRequest,
) -> ShoppingRecommendationResponse:
    # 추천 개수는 최소 1개, 최대 20개로 보정해서 과도한 LLM 호출을 막음
    limit = _normalize_limit(request.limit)
    settings = get_settings()

    # 로컬 개발이나 테스트에서 OPENAI_API_KEY가 없으면 LLM을 호출하지 않고 fallback 추천을 반환
    if not settings.openai_api_key:
        return _fallback_response(request.candidate_products, limit, settings.openai_model)

    try:
        # 실제 LLM 호출 지점. 백엔드가 전달한 후보 재료 안에서만 추천하도록 prompt를 구성
        client = OpenAI(api_key=settings.openai_api_key)
        response = client.responses.create(
            model=settings.openai_model,
            input=_build_prompt(request, limit),
        )

        items = _parse_items(response.output_text, request.candidate_products, limit)
        usage = _extract_usage(response, settings.openai_model)
        return ShoppingRecommendationResponse(items=items, usage=usage)
    except Exception:
        # LLM 호출 실패가 Agent 서버 전체 장애로 이어지지 않도록 후보 재료 기반 추천으로 대체
        # 백엔드 연동 시에는 실패 로그 저장 정책과 함께 다시 조정
        return _fallback_response(request.candidate_products, limit, settings.openai_model)


def _normalize_limit(limit: int) -> int:
    # 백엔드에서도 limit 보정을 하지만 Agent 단에서도 한 번 더 방어
    if limit < 1:
        return 5
    if limit > 20:
        return 20
    return limit


def _build_prompt(request: ShoppingRecommendationRequest, limit: int) -> str:
    # LLM에는 필요한 값만 전달. 사전 재료 후보 전체 Entity가 아니라 추천 판단에 필요한 이름/ID만 보냄
    candidate_names = [
        {
            "productId": product.product_id,
            "productCategoryId": product.product_category_id,
            "productName": product.product_name,
        }
        for product in request.candidate_products
    ]

    fridge_names = [product.product_name for product in request.fridge_items]
    shopping_names = [product.product_name for product in request.shopping_items]

    return f"""
너는 냉장고 관리 서비스의 장보기 추천 Agent야.
사용자의 냉장고 보유 재료와 이미 장보기 목록에 있는 재료는 추천하지 마.
반드시 candidateProducts 안에 있는 재료만 추천해.
추천 개수는 최대 {limit}개야.

fridgeItems:
{json.dumps(fridge_names, ensure_ascii=False)}

shoppingItems:
{json.dumps(shopping_names, ensure_ascii=False)}

candidateProducts:
{json.dumps(candidate_names, ensure_ascii=False)}

응답은 설명 없이 JSON 배열만 반환해.
각 원소 형식:
{{
    "productId": 1,
    "productCategoryId": 1,
    "productName": "감자",
    "quantity": "1개",
    "reason": "추천 이유"
}}
"""


def _parse_items(
    output_text: str,
    candidate_products: list[AgentProduct],
    limit: int,
) -> list[ShoppingRecommendationItemResponse]:
    # LLM이 후보에 없는 재료를 임의로 만들어내는 것을 막기 위해 candidate productId 기준으로 다시 검증
    candidate_by_id = {product.product_id: product for product in candidate_products}
    parsed_items = json.loads(output_text)

    items: list[ShoppingRecommendationItemResponse] = []
    for item in parsed_items:
        product_id = int(item["productId"])
        product = candidate_by_id.get(product_id)
        if product is None:
            continue

        items.append(
            ShoppingRecommendationItemResponse(
                productId=product.product_id,
                productCategoryId=product.product_category_id,
                productName=product.product_name,
                quantity=item.get("quantity", "1개"),
                reason=item.get("reason", "냉장고와 장보기 목록을 기준으로 추천되었습니다."),
            )
        )

        if len(items) >= limit:
            break

    return items


def _extract_usage(response, model_name: str) -> LlmUsageResponse:
    # OpenAI 응답의 usage 값을 백엔드 llm_usage_logs 형식으로 변환
    # 비용 계산은 모델별 단가 정책 확정 후 적용하기 위해 현재 0으로 둠
    usage = getattr(response, "usage", None)
    prompt_tokens = getattr(usage, "input_tokens", 0) if usage else 0
    completion_tokens = getattr(usage, "output_tokens", 0) if usage else 0
    total_tokens = getattr(usage, "total_tokens", prompt_tokens + completion_tokens) if usage else 0

    return LlmUsageResponse(
        modelName=model_name,
        promptTokens=prompt_tokens,
        completionTokens=completion_tokens,
        totalTokens=total_tokens,
        estimatedCost=Decimal("0"),
    )


def _fallback_response(
    candidate_products: list[AgentProduct],
    limit: int,
    model_name: str,
) -> ShoppingRecommendationResponse:
    # LLM 없이도 백엔드-프론트 흐름을 개발할 수 있도록 후보 재료 앞에서부터 추천
    items = [
        ShoppingRecommendationItemResponse(
            productId=product.product_id,
            productCategoryId=product.product_category_id,
            productName=product.product_name,
            quantity="1개",
            reason="LLM 호출 전 또는 실패 시 후보 재료 기준으로 추천되었습니다.",
        )
        for product in candidate_products[:limit]
    ]

    usage = LlmUsageResponse(
        modelName=model_name,
        promptTokens=0,
        completionTokens=0,
        totalTokens=0,
        estimatedCost=Decimal("0"),
    )

    return ShoppingRecommendationResponse(items=items, usage=usage)
