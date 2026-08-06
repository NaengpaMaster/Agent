from decimal import Decimal

from pydantic import BaseModel, Field


class AgentProduct(BaseModel):
    # 백엔드 Product 정보를 Agent가 이해하기 위한 최소 재료 정보
    product_id: int = Field(alias="productId")
    product_category_id: int = Field(alias="productCategoryId")
    product_name: str = Field(alias="productName")


class ShoppingRecommendationRequest(BaseModel):
    # limit: 추천 개수
    # favoriteFoods: 회원 프로필에 등록된 선호 음식 기준
    # fridgeItems: 이미 냉장고에 있는 재료
    # shoppingItems: 이미 장보기 목록에 있는 재료
    # candidateProducts: 백엔드가 추천 후보로 허용한 활성 사전 재료
    limit: int = 5
    favorite_foods: list[str] = Field(default_factory=list, alias="favoriteFoods")
    fridge_items: list[AgentProduct] = Field(default_factory=list, alias="fridgeItems")
    shopping_items: list[AgentProduct] = Field(default_factory=list, alias="shoppingItems")
    candidate_products: list[AgentProduct] = Field(default_factory=list, alias="candidateProducts")


class ShoppingRecommendationItemResponse(BaseModel):
    # 백엔드의 ShoppingRecommendationItemResponse와 맞춰서 반환하는 추천 항목
    product_id: int = Field(alias="productId")
    product_category_id: int = Field(alias="productCategoryId")
    product_name: str = Field(alias="productName")
    quantity: str
    reason: str


class LlmUsageResponse(BaseModel):
    # 백엔드 llm_usage_logs 저장에 사용할 LLM 사용량 정보
    model_name: str = Field(alias="modelName")
    prompt_tokens: int = Field(alias="promptTokens")
    completion_tokens: int = Field(alias="completionTokens")
    total_tokens: int = Field(alias="totalTokens")
    estimated_cost: Decimal = Field(alias="estimatedCost")


class ShoppingRecommendationResponse(BaseModel):
    # items는 사용자에게 보여줄 추천 결과, usage는 백엔드 로그 저장용 정보
    items: list[ShoppingRecommendationItemResponse]
    usage: LlmUsageResponse
