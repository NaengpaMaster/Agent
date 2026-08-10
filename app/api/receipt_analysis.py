from fastapi import APIRouter

from app.schemas.receipt_analysis import (
    ReceiptAnalyzeRequest,
    ReceiptAnalyzeResponse,
)
from app.services.receipt_ocr_service import analyze_receipt

router = APIRouter(prefix="/agent/v1/receipts", tags=["Receipt Agent"])


@router.post("/analyze", response_model=ReceiptAnalyzeResponse)
def analyze_receipt_image(
    request: ReceiptAnalyzeRequest,
) -> ReceiptAnalyzeResponse:
    return analyze_receipt(request)