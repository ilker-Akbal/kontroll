from __future__ import annotations

from collections import deque
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import time

import cv2

from HizTespiti.yolo.src.simple_tracker import Track

from .speed_config import EvidenceConfig


@dataclass
class SpeedViolationEvent:
    camera_id: str
    frame_idx: int
    timestamp_sec: float
    track_id: int
    vehicle_class: str
    speed_kmh: float
    speed_limit_kmh: float
    tolerance_kmh: float
    threshold_kmh: float
    box_xyxy: list[int]
    snapshot_path: str | None
    clip_path: str | None
    created_at: float


class FrameBuffer:
    def __init__(self, max_frames: int):
        self.frames = deque(maxlen=max_frames)

    def add(self, frame_idx: int, frame):
        self.frames.append((frame_idx, frame.copy()))

    def get_from(self, start_frame: int):
        return [(idx, fr) for idx, fr in self.frames if idx >= start_frame]


class EvidenceWriter:
    def __init__(
        self,
        output_dir: str | Path,
        camera_id: str,
        cfg: EvidenceConfig,
        fps: float,
    ):
        self.output_dir = Path(output_dir)
        self.camera_id = camera_id
        self.cfg = cfg
        self.fps = max(float(fps), 1.0)

        self.event_dir = self.output_dir / "events"
        self.snapshot_dir = self.output_dir / "snapshots"
        self.clip_dir = self.output_dir / "clips"

        self.event_dir.mkdir(parents=True, exist_ok=True)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.clip_dir.mkdir(parents=True, exist_ok=True)

        self.events_path = self.event_dir / f"{camera_id}_speed_violations.jsonl"

    def _stamp(self) -> str:
        return time.strftime("%Y%m%d_%H%M%S")

    def save_event(
        self,
        frame_idx: int,
        timestamp_sec: float,
        frame_vis,
        track: Track,
        speed_kmh: float,
        speed_limit_kmh: float,
        tolerance_kmh: float,
        threshold_kmh: float,
        frame_buffer: FrameBuffer,
    ) -> SpeedViolationEvent:
        stamp = self._stamp()
        base_name = f"{self.camera_id}_track{track.track_id}_{int(speed_kmh)}kmh_{stamp}_f{frame_idx}"

        snapshot_path = None
        clip_path = None

        if self.cfg.save_snapshot:
            snapshot_path = str(self.snapshot_dir / f"{base_name}.jpg")
            cv2.imwrite(
                snapshot_path,
                frame_vis,
                [int(cv2.IMWRITE_JPEG_QUALITY), int(self.cfg.jpeg_quality)],
            )

        if self.cfg.save_clip:
            start_frame = int(frame_idx - self.cfg.clip_pre_sec * self.fps)
            frames = frame_buffer.get_from(start_frame)

            if frames:
                h, w = frames[0][1].shape[:2]
                clip_path = str(self.clip_dir / f"{base_name}.mp4")
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(clip_path, fourcc, self.fps, (w, h))

                for _, fr in frames:
                    writer.write(fr)

                writer.write(frame_vis)
                writer.release()

        x1, y1, x2, y2 = [int(v) for v in track.box]

        event = SpeedViolationEvent(
            camera_id=self.camera_id,
            frame_idx=frame_idx,
            timestamp_sec=timestamp_sec,
            track_id=track.track_id,
            vehicle_class=track.cls_name,
            speed_kmh=float(speed_kmh),
            speed_limit_kmh=float(speed_limit_kmh),
            tolerance_kmh=float(tolerance_kmh),
            threshold_kmh=float(threshold_kmh),
            box_xyxy=[x1, y1, x2, y2],
            snapshot_path=snapshot_path,
            clip_path=clip_path,
            created_at=time.time(),
        )

        with self.events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(event), ensure_ascii=False) + "\n")

        return event