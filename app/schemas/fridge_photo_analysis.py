from pydantic import BaseModel, Field

from app.schemas.shopping_recommendation import LlmUsageResponse


class FridgePhotoAnalyzeRequest(BaseModel):
    fridge_photo_analysis_id: int = Field(alias="fridgePhotoAnalysisId")
    image_base64: str = Field(alias="imageBase64")
    mime_type: str = Field(alias="mimeType")


class FridgePhotoAnalyzeItemResponse(BaseModel):
    name: str
    quantity: str = "1개"


class FridgePhotoAnalyzeResponse(BaseModel):
    raw_text: str = Field(alias="rawText")
    items: list[FridgePhotoAnalyzeItemResponse]
    usage: LlmUsageResponse
