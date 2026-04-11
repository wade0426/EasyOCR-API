from __future__ import annotations

from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    MODEL_STORAGE_DIR: Path = Field(default=Path(r"C:\Users\ramune\Desktop\LocalModels\easyocr"))
    OCR_LANGUAGES: list[str] = Field(default_factory=lambda: ["ch_tra", "en"])
    USE_GPU: bool = False
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    LOG_LEVEL: str = "info"

    @field_validator("MODEL_STORAGE_DIR", mode="before")
    @classmethod
    def _coerce_model_path(cls, value: str | Path) -> Path:
        return Path(value)

    def validate_model_storage_dir(self) -> None:
        if not self.MODEL_STORAGE_DIR.exists() or not self.MODEL_STORAGE_DIR.is_dir():
            raise ValueError(
                f"MODEL_STORAGE_DIR does not exist or is not a directory: {self.MODEL_STORAGE_DIR}"
            )


settings = Settings()
