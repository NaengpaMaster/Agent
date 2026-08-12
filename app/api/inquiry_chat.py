from secrets import compare_digest
from typing import Optional

from fastapi import APIRouter, Header, HTTPException, status

from app.core.config import get_settings
from app.schemas.inquiry_chat import InquiryChatRequest, InquiryChatResponse
from app.services.inquiry_chat_service import answer_inquiry

router = APIRouter(prefix="/agent/v1/inquiry-chat", tags=["Inquiry Chat Agent"])


@router.post("/answers", response_model=InquiryChatResponse)
def create_inquiry_chat_answer(
    request: InquiryChatRequest,
    x_agent_api_key: Optional[str] = Header(default=None),
) -> InquiryChatResponse:
    expected_key = get_settings().agent_api_key
    if not x_agent_api_key or not compare_digest(x_agent_api_key, expected_key):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="인증되지 않은 Agent 요청입니다.")
    return answer_inquiry(request)
