from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from config import settings
from routers.chat import chat_router
from services.ocr_service import get_ocr_service

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        settings.validate_model_storage_dir()
        ocr_service = get_ocr_service()
        ocr_service.load()
        logger.info("Service startup complete. OCR model path: %s", settings.MODEL_STORAGE_DIR)
    except Exception as exc:
        logger.critical("Failed to start service: %s", exc)
        sys.exit(1)

    yield

    logger.info("Service shutdown complete.")


app = FastAPI(
    title="EasyOCR Vision API",
    description="OpenAI Chat Completions (Vision) compatible API powered by EasyOCR",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(chat_router, prefix="/v1")


@app.get("/health")
async def health() -> dict[str, bool | str]:
    ocr_service = get_ocr_service()
    return {"status": "ok", "model_loaded": ocr_service.loaded}


if __name__ == "__main__":
    uvicorn.run("main:app", host=settings.HOST, port=settings.PORT, log_level=settings.LOG_LEVEL)
