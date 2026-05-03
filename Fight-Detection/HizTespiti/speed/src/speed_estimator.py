from __future__ import annotations

from dataclasses import dataclass
import math

from HizTespiti.yolo.src.simple_tracker import Track

from .speed_config import SpeedConfig


@dataclass
class SpeedResult:
    track_id: int
    speed_kmh: float | None
    valid: bool
    reason: str
    points_used: int


class SpeedEstimator:
    def __init__(self, cfg: SpeedConfig, meter_per_pixel: float | None, fps: float):
        self.cfg = cfg
        self.meter_per_pixel = meter_per_pixel
        self.fps = max(float(fps), 1.0)

    def estimate(self, track: Track) -> SpeedResult:
        if self.meter_per_pixel is None or self.meter_per_pixel <= 0:
            return SpeedResult(track.track_id, None, False, "missing_scale", 0)

        hist = track.history[-self.cfg.smooth_window:]

        if len(hist) < self.cfg.min_track_points:
            return SpeedResult(track.track_id, None, False, "not_enough_points", len(hist))

        f1, x1, y1 = hist[0]
        f2, x2, y2 = hist[-1]

        frame_delta = max(1, int(f2 - f1))
        time_delta = frame_delta / self.fps

        if time_delta < self.cfg.min_time_delta_sec:
            return SpeedResult(track.track_id, None, False, "time_delta_too_low", len(hist))

        if time_delta > self.cfg.max_time_delta_sec:
            return SpeedResult(track.track_id, None, False, "time_delta_too_high", len(hist))

        pixel_dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        meter_dist = pixel_dist * self.meter_per_pixel

        speed_mps = meter_dist / max(time_delta, 1e-6)
        speed_kmh = speed_mps * 3.6

        if speed_kmh < self.cfg.min_valid_speed_kmh:
            return SpeedResult(track.track_id, speed_kmh, False, "speed_too_low", len(hist))

        if speed_kmh > self.cfg.max_valid_speed_kmh:
            return SpeedResult(track.track_id, speed_kmh, False, "speed_too_high", len(hist))

        return SpeedResult(track.track_id, speed_kmh, True, "ok", len(hist))