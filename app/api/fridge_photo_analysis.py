from fastapi import APIRouter

from app.schemas.fridge_photo_analysis import (
    FridgePhotoAnalyzeRequest,
    FridgePhotoAnalyzeResponse,
)
from app.services.fridge_photo_analysis_service import analyze_fridge_photo

router = APIRouter(prefix="/agent/v1/fridge-photos", tags=["Fridge Photo Agent"])


@router.post("/analyze", response_model=FridgePhotoAnalyzeResponse)
def analyze_fridge_photo_image(
    request: FridgePhotoAnalyzeRequest,
) -> FridgePhotoAnalyzeResponse:
    return analyze_fridge_photo(request)
