"""Application settings and validation."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class RuntimeSettings(BaseModel):
    esp32_url: str
    piece_model_path: str
    piece_conf: float = Field(ge=0.1, le=1.0)
    smoothing_frames: int = Field(ge=2, le=30)
    warp_size: int = Field(ge=400, le=1600)
    min_board_area_px: int = Field(gt=0)
    min_frame_width: int = Field(gt=0)
    min_frame_height: int = Field(gt=0)
    rotate_180: bool = False
    exports_dir: str
    failures_dir: str
    opening_book_path: str | None = None
    show_square_labels: bool = True
    show_confidence_scores: bool = True
    show_piece_count_hud: bool = True
    highlight_last_move: bool = True
    hud_font_scale: float = Field(gt=0.1, le=2.0)
    grid_opacity: float = Field(ge=0.0, le=1.0)
    highlight_opacity: float = Field(ge=0.0, le=1.0)

    @field_validator("esp32_url")
    @classmethod
    def validate_stream_url(cls, value: str) -> str:
        parsed = urlparse(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("esp32_url must be a valid HTTP URL.")
        if not (parsed.path.endswith("/stream") or parsed.path.endswith("/video")):
            raise ValueError("esp32_url must end with /stream or /video.")
        return value

    @model_validator(mode="after")
    def validate_warp(self) -> "RuntimeSettings":
        if self.warp_size % 8 != 0:
            raise ValueError("warp_size must be divisible by 8.")
        return self


class EnvSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    SUPABASE_URL: str
    SUPABASE_KEY: str

    @field_validator("SUPABASE_URL")
    @classmethod
    def validate_supabase_url(cls, value: str) -> str:
        parsed = urlparse(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("SUPABASE_URL must be a valid URL.")
        return value


class AppSettings(BaseModel):
    runtime: RuntimeSettings
    env: EnvSettings


def load_settings(config_path: str = "config.yaml") -> AppSettings:
    """Load and validate settings from config + environment."""
    load_dotenv()
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    data = yaml.safe_load(config_file.read_text(encoding="utf-8")) or {}
    runtime = RuntimeSettings(**data)
    env = EnvSettings()

    if not Path(runtime.piece_model_path).exists():
        raise FileNotFoundError(f"Model file does not exist: {runtime.piece_model_path}")

    exports_dir = Path(runtime.exports_dir)
    failures_dir = Path(runtime.failures_dir)
    exports_dir.mkdir(parents=True, exist_ok=True)
    failures_dir.mkdir(parents=True, exist_ok=True)

    if runtime.opening_book_path:
        if not Path(runtime.opening_book_path).exists():
            raise FileNotFoundError(f"Opening book file does not exist: {runtime.opening_book_path}")

    return AppSettings(runtime=runtime, env=env)

