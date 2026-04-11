from __future__ import annotations

import logging
from typing import Any, NoReturn

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from schemas.openai_compat import (
    AssistantMessage,
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    OpenAIErrorBody,
    OpenAIErrorResponse,
    UsageInfo,
)
from services.ocr_service import get_ocr_service
from utils.image_utils import bytes_to_numpy, extract_first_image_url, parse_data_url

logger = logging.getLogger(__name__)

chat_router = APIRouter(tags=["chat"])


class _OpenAIError(Exception):
    def __init__(self, status_code: int, payload: dict[str, Any]) -> None:
        super().__init__(payload.get("error", {}).get("message", "OpenAI error"))
        self.status_code = status_code
        self.payload = payload


def _raise_openai_error(status_code: int, message: str) -> NoReturn:
    payload = OpenAIErrorResponse(error=OpenAIErrorBody(message=message))
    raise _OpenAIError(status_code=status_code, payload=payload.model_dump())


@chat_router.post("/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest) -> ChatCompletionResponse | JSONResponse:
    try:
        if request.stream:
            _raise_openai_error(400, "Streaming is not supported by this server.")

        image_url = extract_first_image_url(request.messages)
        if image_url is None:
            _raise_openai_error(400, "No image found in the request.")

        try:
            _, raw_bytes = parse_data_url(image_url)
        except ValueError as exc:
            _raise_openai_error(400, str(exc))

        try:
            image = bytes_to_numpy(raw_bytes)
        except ValueError as exc:
            _raise_openai_error(400, str(exc))

        ocr_service = get_ocr_service()
        try:
            ocr_text = ocr_service.recognize(image)
        except Exception:
            logger.exception("OCR processing failed.")
            _raise_openai_error(500, "OCR processing failed.")

        completion_tokens = len(ocr_text.split())
        usage = UsageInfo(
            prompt_tokens=0,
            completion_tokens=completion_tokens,
            total_tokens=completion_tokens,
        )

        return ChatCompletionResponse(
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=AssistantMessage(role="assistant", content=ocr_text),
                    finish_reason="stop",
                )
            ],
            usage=usage,
        )
    except _OpenAIError as exc:
        return JSONResponse(status_code=exc.status_code, content=exc.payload)
