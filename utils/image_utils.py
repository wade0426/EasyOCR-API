from __future__ import annotations

import base64
import io
import re

import numpy as np
from PIL import Image, UnidentifiedImageError

from schemas.openai_compat import Message


SUPPORTED_MIME_TYPES = {
    "image/png",
    "image/jpeg",
    "image/jpg",
    "image/webp",
    "image/bmp",
}

DATA_URL_PATTERN = re.compile(r"^data:(?P<mime>[a-zA-Z0-9.+-]+/[a-zA-Z0-9.+-]+);base64,(?P<data>.+)$")


def parse_data_url(data_url: str) -> tuple[str, bytes]:
    if not data_url.startswith("data:"):
        raise ValueError("Only data URL (base64) images are supported.")

    match = DATA_URL_PATTERN.match(data_url)
    if match is None:
        raise ValueError("Invalid data URL format. Expected: data:<mime>;base64,<data>")

    mime_type = match.group("mime").lower()
    if mime_type not in SUPPORTED_MIME_TYPES:
        raise ValueError(f"Unsupported image format: {mime_type}")

    encoded = match.group("data")
    try:
        raw_bytes = base64.b64decode(encoded, validate=True)
    except Exception as exc:
        raise ValueError("Invalid base64 image data.") from exc

    return mime_type, raw_bytes


def bytes_to_numpy(raw_bytes: bytes) -> np.ndarray:
    try:
        with Image.open(io.BytesIO(raw_bytes)) as image:
            rgb_image = image.convert("RGB")
            return np.array(rgb_image, dtype=np.uint8)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError("Cannot decode image data.") from exc


def extract_first_image_url(messages: list[Message]) -> str | None:
    for message in messages:
        if isinstance(message.content, str):
            continue

        for content_part in message.content:
            if content_part.type == "image_url":
                return content_part.image_url.url

    return None
