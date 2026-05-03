from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any
import time


@dataclass
class RoadRoi:
    enabled: bool = False
    polygon: list[list[int]] = field(default_factory=list)


@dataclass
class MeasurementConfig:
    mode: str = "pixel_scale"
    direction: str = "AUTO"
    line_a: list[list[int]] = field(default_factory=list)
    line_b: list[list[int]] = field(default_factory=list)
    distance_m: float | None = None


@dataclass
class ScaleConfig:
    source: str = "auto_vehicle_size"
    meter_per_pixel: float | None = None
    confidence: float = 0.0
    user_corrected: bool = False


@dataclass
class CameraCalibration:
    camera_id: str
    speed_limit_kmh: float = 30.0
    tolerance_kmh: float = 5.0
    road_roi: RoadRoi = field(default_factory=RoadRoi)
    measurement: MeasurementConfig = field(default_factory=MeasurementConfig)
    scale: ScaleConfig = field(default_factory=ScaleConfig)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)