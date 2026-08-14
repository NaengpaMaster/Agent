import json
import logging
import time
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

GPT_4_1_MINI_MODEL = "gpt-4.1-mini"
GPT_4_1_MINI_INPUT_PRICE_PER_1M = Decimal("0.40")
GPT_4_1_MINI_OUTPUT_PRICE_PER_1M = Decimal("1.60")
TOKENS_PER_MILLION = Decimal("1000000")
logger = logging.getLogger("uvicorn.error")


def recommend_shopping_items(
    request: ShoppingRecommendationRequest,
) -> ShoppingRecommendationResponse:
    # 추천 개수는 최소 1개, 최대 20개로 보정해서 과도한 LLM 호출을 막음
    limit = _normalize_limit(request.limit)
    settings = get_settings()

    # 로컬 개발이나 테스트에서 OPENAI_API_KEY가 없으면 LLM을 호출하지 않고 fallback 추천을 반환
    if not settings.openai_api_key:
        return _fallback_response(
            request.candidate_products,
            request.favorite_foods,
            limit,
            settings.openai_model,
        )

    try:
        # 실제 LLM 호출 지점. 백엔드가 전달한 후보 재료 안에서만 추천하도록 prompt를 구성
        client = OpenAI(api_key=settings.openai_api_key)
        started_at = time.perf_counter()
        try:
            response = client.responses.create(
                model=settings.openai_model,
                input=_build_prompt(request, limit),
            )
        finally:
            logger.info(
                "shopping OpenAI call elapsed_ms=%.1f candidate_count=%d limit=%d",
                (time.perf_counter() - started_at) * 1000,
                len(request.candidate_products),
                limit,
            )

        items = _parse_items(response.output_text, request.candidate_products, limit)
        usage = _extract_usage(response, settings.openai_model)
        return ShoppingRecommendationResponse(items=items, usage=usage)
    except Exception:
        # LLM 호출 실패가 Agent 서버 전체 장애로 이어지지 않도록 후보 재료 기반 추천으로 대체
        # 백엔드 연동 시에는 실패 로그 저장 정책과 함께 다시 조정
        return _fallback_response(
            request.candidate_products,
            request.favorite_foods,
            limit,
            settings.openai_model,
        )


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
    favorite_foods = request.favorite_foods

    return f"""
너는 냉장고 관리 서비스의 장보기 추천 Agent야.
사용자의 냉장고 보유 재료와 이미 장보기 목록에 있는 재료는 추천하지 마.
사용자의 선호 음식과 어울리는 재료를 우선 추천해.
반드시 candidateProducts 안에 있는 재료만 추천해.
candidateProducts 밖의 재료는 절대 새로 만들지 마.
추천 개수는 최대 {limit}개야.

favoriteFoods:
{json.dumps(favorite_foods, ensure_ascii=False)}

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
    "reason": "선호 음식 또는 부족 재료 기준을 포함한 짧은 추천 이유"
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
    usage = getattr(response, "usage", None)
    prompt_tokens = getattr(usage, "input_tokens", 0) if usage else 0
    completion_tokens = getattr(usage, "output_tokens", 0) if usage else 0
    total_tokens = getattr(usage, "total_tokens", prompt_tokens + completion_tokens) if usage else 0

    return LlmUsageResponse(
        modelName=model_name,
        promptTokens=prompt_tokens,
        completionTokens=completion_tokens,
        totalTokens=total_tokens,
        estimatedCost=_calculate_estimated_cost(model_name, prompt_tokens, completion_tokens),
    )


def _calculate_estimated_cost(
    model_name: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> Decimal:
    # 현재 운영 모델인 gpt-4.1-mini 기준 예상 비용을 계산한다.
    # 다른 모델을 쓰면 단가 정책 확정 전까지 0으로 기록한다.
    if model_name != GPT_4_1_MINI_MODEL:
        return Decimal("0")

    input_cost = (
        Decimal(prompt_tokens)
        * GPT_4_1_MINI_INPUT_PRICE_PER_1M
        / TOKENS_PER_MILLION
    )
    output_cost = (
        Decimal(completion_tokens)
        * GPT_4_1_MINI_OUTPUT_PRICE_PER_1M
        / TOKENS_PER_MILLION
    )

    return input_cost + output_cost


def _fallback_response(
    candidate_products: list[AgentProduct],
    favorite_foods: list[str],
    limit: int,
    model_name: str,
) -> ShoppingRecommendationResponse:
    # LLM 없이도 백엔드-프론트 흐름을 개발할 수 있도록 후보 재료 앞에서부터 추천
    reason = _build_fallback_reason(favorite_foods)

    items = [
        ShoppingRecommendationItemResponse(
            productId=product.product_id,
            productCategoryId=product.product_category_id,
            productName=product.product_name,
            quantity="1개",
            reason=reason,
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


def _build_fallback_reason(favorite_foods: list[str]) -> str:
    # OpenAI 키가 없거나 호출 실패 시에도 추천 기준이 사용자에게 모호하게 보이지 않도록 이유를 고정
    if not favorite_foods:
        return "냉장고와 장보기 목록에 없고 못 먹는 재료가 아닌 후보 재료입니다."

    return f"선호 음식({', '.join(favorite_foods)})과 못 먹는 재료 제외 기준으로 추천되었습니다."
