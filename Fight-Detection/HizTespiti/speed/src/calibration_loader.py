from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json


@dataclass
class LoadedCalibration:
    camera_id: str
    speed_limit_kmh: float
    tolerance_kmh: float
    meter_per_pixel: float | None
    scale_confidence: float
    user_corrected: bool
    raw: dict[str, Any]


def load_calibration(path: str | Path) -> LoadedCalibration:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Kalibrasyon dosyası bulunamadı: {path}\n"
            f"Önce şunu çalıştır:\n"
            f"python -m HizTespiti.calibration.run_calibration"
        )

    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    scale = raw.get("scale", {}) or {}

    mpp = scale.get("meter_per_pixel", None)

    if mpp is not None:
        mpp = float(mpp)

    return LoadedCalibration(
        camera_id=str(raw.get("camera_id", "cam_001")),
        speed_limit_kmh=float(raw.get("speed_limit_kmh", 30)),
        tolerance_kmh=float(raw.get("tolerance_kmh", 5)),
        meter_per_pixel=mpp,
        scale_confidence=float(scale.get("confidence", 0.0)),
        user_corrected=bool(scale.get("user_corrected", False)),
        raw=raw,
    )