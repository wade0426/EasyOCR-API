from __future__ import annotations

import logging
from importlib import import_module
from typing import Any

import numpy as np

from config import settings

logger = logging.getLogger(__name__)

def _build_reader(*, languages: list[str], model_storage_directory: str, gpu: bool) -> Any:
    try:
        easyocr_module = import_module("easyocr")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "easyocr is not installed. Please install dependencies from requirements.txt before starting the service."
        ) from exc

    return easyocr_module.Reader(
        lang_list=languages,
        model_storage_directory=model_storage_directory,
        gpu=gpu,
        download_enabled=False,
    )


class OCRService:
    def __init__(self, model_storage_directory: str, languages: list[str], gpu: bool) -> None:
        self.model_storage_directory: str = model_storage_directory
        self.languages: list[str] = languages
        self.gpu: bool = gpu
        self.reader: Any | None = None
        self._loaded: bool = False

    @property
    def loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        try:
            self.reader = _build_reader(
                languages=self.languages,
                model_storage_directory=self.model_storage_directory,
                gpu=self.gpu,
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to initialize EasyOCR reader: {exc}") from exc

        self._loaded = True
        logger.info(
            "EasyOCR models loaded successfully from %s (languages=%s, gpu=%s)",
            self.model_storage_directory,
            self.languages,
            self.gpu,
        )

    def recognize(self, image: np.ndarray) -> str:
        if self.reader is None:
            raise RuntimeError("OCR service is not loaded.")

        results: list[str] = self.reader.readtext(image, detail=0)
        if not results:
            return ""

        return "\n".join(results)


_ocr_service: OCRService | None = None


def get_ocr_service() -> OCRService:
    global _ocr_service

    if _ocr_service is None:
        _ocr_service = OCRService(
            model_storage_directory=str(settings.MODEL_STORAGE_DIR),
            languages=settings.OCR_LANGUAGES,
            gpu=settings.USE_GPU,
        )

    return _ocr_service
