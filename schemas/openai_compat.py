from __future__ import annotations

import time
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


class ImageUrlContent(BaseModel):
    url: str


class ContentPartText(BaseModel):
    type: Literal["text"]
    text: str


class ContentPartImage(BaseModel):
    type: Literal["image_url"]
    image_url: ImageUrlContent


ContentPart = ContentPartText | ContentPartImage


class Message(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str | list[ContentPart]


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[Message]
    max_tokens: int | None = None
    temperature: float | None = None
    stream: bool | None = False

    @field_validator("messages")
    @classmethod
    def messages_must_not_be_empty(cls, value: list[Message]) -> list[Message]:
        if len(value) == 0:
            raise ValueError("messages must not be empty")
        return value


class AssistantMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str


class ChatCompletionChoice(BaseModel):
    index: int
    message: AssistantMessage
    finish_reason: Literal["stop"] = "stop"


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid4().hex[:8]}")
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionChoice]
    usage: UsageInfo


class OpenAIErrorBody(BaseModel):
    message: str
    type: str = "invalid_request_error"
    code: str | None = None


class OpenAIErrorResponse(BaseModel):
    error: OpenAIErrorBody
