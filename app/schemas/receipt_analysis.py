from pydantic import BaseModel, Field

from app.schemas.shopping_recommendation import LlmUsageResponse

class ReceiptAnalyzeRequest(BaseModel):
    receipt_analysis_id: int = Field(alias="receiptAnalysisId")
    image_base64: str = Field(alias="imageBase64")
    mime_type: str = Field(alias="mimeType")

class ReceiptAnalyzeItemResponse(BaseModel):
    name: str
    quantity: str = "1개"

class ReceiptAnalyzeResponse(BaseModel):
    raw_text: str = Field(alias="rawText")
    items: list[ReceiptAnalyzeItemResponse]
    usage: LlmUsageResponse
