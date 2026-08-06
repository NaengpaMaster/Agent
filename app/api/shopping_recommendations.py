from fastapi import APIRouter

from app.schemas.shopping_recommendation import (
    ShoppingRecommendationRequest,
    ShoppingRecommendationResponse,
)
from app.services.shopping_recommendation_service import recommend_shopping_items

router = APIRouter(prefix="/agent/v1/shopping", tags=["Shopping Agent"])


@router.post("/recommendations", response_model=ShoppingRecommendationResponse)
def create_shopping_recommendations(
    request: ShoppingRecommendationRequest,
) -> ShoppingRecommendationResponse:
    # 백엔드가 전달한 냉장고/장보기/후보 재료 데이터를 받아 장보기 추천 결과를 생성
    # Agent는 DB를 직접 수정하지 않고, 추천 결과만 백엔드에 반환
    return recommend_shopping_items(request)
