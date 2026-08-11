from typing import Literal

from pydantic import BaseModel, Field

from app.schemas.shopping_recommendation import LlmUsageResponse


class InquiryChatHistoryMessage(BaseModel):
    role: Literal["USER", "ASSISTANT"]
    content: str = Field(min_length=1, max_length=2000)


class InquiryKnowledgeContext(BaseModel):
    source_name: str = Field(alias="sourceName")
    content: str = Field(min_length=1)


class InquiryChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    history: list[InquiryChatHistoryMessage] = Field(default_factory=list)
    contexts: list[InquiryKnowledgeContext] = Field(default_factory=list)


class InquiryChatResponse(BaseModel):
    answer: str
    answerable: bool
    sources: list[str]
    usage: LlmUsageResponse
