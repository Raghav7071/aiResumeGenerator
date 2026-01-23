from pydantic import BaseModel, Field
from typing import List


class ProcessRequest(BaseModel):
    filename: str


class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=20)


class SearchResult(BaseModel):
    text: str
    score: float
    metadata: dict


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    history: List[dict] = []


class ChatResponse(BaseModel):
    answer: str
    citations: List[dict] = []
