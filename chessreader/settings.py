"""Config loading for the chess reader."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator

PROJECT_DIR = Path(__file__).resolve().parent.parent  # the repo root; relative config paths resolve here
DEFAULT_CONFIG = PROJECT_DIR / "config.yaml"


def resolve_path(path: str | Path) -> Path:
    """Resolve a config path against the project root."""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_DIR / p


class CameraSettings(BaseModel):
    host: str = "192.168.4.1"
    http_port: int = 80
    stream_port: int = 81
    source: Literal["capture", "stream"] = "capture"
    framesize: str = "SVGA"
    quality: int = Field(10, ge=4, le=63)
    vflip: bool = False
    hmirror: bool = False
    extra_controls: dict[str, int] = Field(default_factory=dict)
    apply_controls: bool = True
    connect_timeout_s: float = Field(3.0, gt=0)
    read_timeout_s: float = Field(5.0, gt=0)
    reconnect_backoff_s: float = Field(1.0, gt=0)
    reconnect_backoff_max_s: float = Field(10.0, gt=0)

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.http_port}"

    @property
    def stream_url(self) -> str:
        return f"http://{self.host}:{self.stream_port}/stream"

    @property
    def capture_url(self) -> str:
        return f"{self.base_url}/capture"


class CalibrationSettings(BaseModel):
    dir: str = "calib"
    warp_size: int = Field(800, ge=400, le=1600)
    detect_attempts: int = Field(5, ge=1)
    max_reprojection_px: float = Field(4.0, gt=0)
    canny_low: int = 50
    canny_high: int = 150

    @field_validator("warp_size")
    @classmethod
    def divisible_by_8(cls, value: int) -> int:
        if value % 8 != 0:
            raise ValueError("warp_size must be divisible by 8")
        return value


class ViewSettings(BaseModel):
    show_labels: bool = True
    captures_dir: str = "captures"


class RecordSettings(BaseModel):
    dir: str = "recordings"
    fps: float = Field(10.0, gt=0)


class LoggingSettings(BaseModel):
    dir: str = "logs"
    level: str = "INFO"


class GateSettings(BaseModel):
    pixel_threshold: float = 18.0      # gray-level change that counts as motion
    motion_fraction: float = 0.004     # share of board pixels that must change to count as motion
    stable_time_s: float = 0.8         # board must be still this long ...
    min_stable_frames: int = 2         # ... and for at least this many frames


class ReaderSettings(BaseModel):
    scale: float = Field(0.5, gt=0.1, le=1.0)   # work resolution for change maps
    t_low: float = 10.0                # color distance below which a pixel is unchanged
    t_high: float = 28.0               # color distance above which a pixel surely changed
    q_in: float = 0.55                 # chance a pixel inside a moved piece's outline visibly changes
    q_out: float = 0.03                # noise rate elsewhere
    height_scale: float = 1.0          # scales the assumed piece heights
    min_change_px: float = 900.0       # less observed change than this = nothing happened
    occlusion_fraction: float = 0.35   # more than this share of the board changed = hand/arm in view
    unexplained_alarm_px: float = 6000.0  # change no move explains: ignore below this, ask above


class ConfirmTiers(BaseModel):
    min_score: float = 1000.0
    t1_margin: float = 1500.0
    t2_margin: float = 800.0
    max_unexplained: float = 0.35
    vote_frames: int = 3


class LlmSettings(BaseModel):
    enabled: bool = False              # tier 3; needs ANTHROPIC_API_KEY in .env
    model: str = "claude-sonnet-5-5"


class HealthSettings(BaseModel):
    sync_every_moves: int = 10
    sync_min_confidence: float = 0.9
    drift_every_s: float = 10.0
    drift_threshold: float = 0.45


class EngineSettings(BaseModel):
    enabled: bool = True
    path: str = ""
    think_time_s: float = 0.5
    multipv: int = 3
    show_best_moves: bool = True       # false = training/review mode: hide best moves while playing


class StorageSettings(BaseModel):
    exports_dir: str = "exports"
    sessions_dir: str = "sessions"
    record_frames: bool = True         # keep every frame for replay testing
    white: str = "White"
    black: str = "Black"


class PiecesSettings(BaseModel):
    weights: str = "models/pieces.pt"
    conf: float = 0.15
    use_for_promotion: bool = True


class Settings(BaseModel):
    camera: CameraSettings = Field(default_factory=CameraSettings)
    calibration: CalibrationSettings = Field(default_factory=CalibrationSettings)
    gate: GateSettings = Field(default_factory=GateSettings)
    reader: ReaderSettings = Field(default_factory=ReaderSettings)
    confirm: ConfirmTiers = Field(default_factory=ConfirmTiers)
    llm: LlmSettings = Field(default_factory=LlmSettings)
    health: HealthSettings = Field(default_factory=HealthSettings)
    engine: EngineSettings = Field(default_factory=EngineSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    pieces: PiecesSettings = Field(default_factory=PiecesSettings)
    view: ViewSettings = Field(default_factory=ViewSettings)
    record: RecordSettings = Field(default_factory=RecordSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)


def load_settings(config_path: str | Path = DEFAULT_CONFIG) -> Settings:
    """Load and validate settings from a YAML file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return Settings(**data)
