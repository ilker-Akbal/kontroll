from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Deque, Dict, List, Optional

import cv2


@dataclass
class Stage3Result:
    camera_id: str
    source: str
    event_id: str
    event_start_ts: float
    event_end_ts: float
    clip_path: str
    fight_prob: float
    fight_label: str
    pose_score_max: float
    pose_score_mean: float


@dataclass
class IncidentSegment:
    result: Stage3Result
    arrived_at: float = field(default_factory=time.time)


@dataclass
class TemporalIncidentState:
    incident_id: str
    camera_id: str
    source: str
    state: str = "idle"  # idle, candidate, confirmed, cooldown
    segments: List[IncidentSegment] = field(default_factory=list)
    recent_scores: Deque[float] = field(default_factory=lambda: deque(maxlen=5))
    recent_is_fight: Deque[int] = field(default_factory=lambda: deque(maxlen=5))
    candidate_since_ts: Optional[float] = None
    confirmed_since_ts: Optional[float] = None
    last_event_end_ts: Optional[float] = None
    last_update_wall_ts: float = field(default_factory=time.time)
    alarm_sent: bool = False
    cooldown_until_ts: float = 0.0

    def add_segment(self, seg: IncidentSegment) -> None:
        self.segments.append(seg)
        self.recent_scores.append(float(seg.result.fight_prob))
        self.recent_is_fight.append(1 if float(seg.result.fight_prob) >= 0.5 else 0)
        self.last_event_end_ts = float(seg.result.event_end_ts)
        self.last_update_wall_ts = time.time()

    @property
    def start_ts(self) -> float:
        if not self.segments:
            return 0.0
        return float(min(s.result.event_start_ts for s in self.segments))

    @property
    def end_ts(self) -> float:
        if not self.segments:
            return 0.0
        return float(max(s.result.event_end_ts for s in self.segments))

    @property
    def duration_sec(self) -> float:
        return max(0.0, float(self.end_ts) - float(self.start_ts))

    @property
    def max_prob(self) -> float:
        if not self.segments:
            return 0.0
        return max(float(s.result.fight_prob) for s in self.segments)

    @property
    def mean_prob(self) -> float:
        if not self.segments:
            return 0.0
        vals = [float(s.result.fight_prob) for s in self.segments]
        return sum(vals) / max(1, len(vals))

    @property
    def part_count(self) -> int:
        return len(self.segments)

    def vote_count(self, thr: float) -> int:
        return sum(1 for x in self.recent_scores if float(x) >= float(thr))

    def last_gap_to(self, result: Stage3Result) -> float:
        if self.last_event_end_ts is None:
            return 0.0
        return float(result.event_start_ts) - float(self.last_event_end_ts)


class IncidentAggregator:
    def __init__(
        self,
        out_dir: str,
        merge_gap_sec: float = 12.0,
        max_bridge_nonfight: int = 2,
        enter_thr: float = 0.62,
        keep_thr: float = 0.48,
        vote_window: int = 5,
        vote_enter_needed: int = 3,
        vote_keep_needed: int = 2,
        min_incident_segments: int = 2,
        single_strong_fight_thr: float = 0.78,
        confirm_min_duration_sec: float = 1.0,
        cooldown_sec: float = 20.0,
        keep_temp_parts: bool = True,
        write_nonfight_incidents: bool = False,
        clip_ready_wait_sec: float = 8.0,
        stale_finalize_sec: float = 10.0,
        temporal_iou_merge_thr: float = 0.50,
        sweep_interval_sec: float = 0.50,
    ):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.incidents_jsonl = self.out_dir.parent / "incidents.jsonl"

        self.merge_gap_sec = float(merge_gap_sec)
        self.max_bridge_nonfight = int(max_bridge_nonfight)
        self.enter_thr = float(enter_thr)
        self.keep_thr = float(keep_thr)
        self.vote_window = int(vote_window)
        self.vote_enter_needed = int(vote_enter_needed)
        self.vote_keep_needed = int(vote_keep_needed)
        self.min_incident_segments = int(min_incident_segments)
        self.single_strong_fight_thr = float(single_strong_fight_thr)
        self.confirm_min_duration_sec = float(confirm_min_duration_sec)
        self.cooldown_sec = float(cooldown_sec)
        self.keep_temp_parts = bool(keep_temp_parts)
        self.write_nonfight_incidents = bool(write_nonfight_incidents)
        self.clip_ready_wait_sec = float(clip_ready_wait_sec)
        self.stale_finalize_sec = float(stale_finalize_sec)
        self.temporal_iou_merge_thr = float(temporal_iou_merge_thr)
        self.sweep_interval_sec = float(sweep_interval_sec)

        self.lock = threading.Lock()
        self.by_camera: Dict[str, TemporalIncidentState] = {}
        self.counter: Dict[str, int] = {}

        self._stop_event = threading.Event()
        self._sweeper = threading.Thread(
            target=self._sweeper_loop,
            name="incident_sweeper",
            daemon=True,
        )
        self._sweeper.start()

    def submit(self, result: Stage3Result):
        with self.lock:
            st = self.by_camera.get(result.camera_id)
            if st is None:
                st = self._new_state(result.camera_id, result.source)
                self.by_camera[result.camera_id] = st

            if st.state == "cooldown":
                if float(result.event_start_ts) <= float(st.cooldown_until_ts):
                    if self._can_merge(st, result):
                        self._append_or_suppress_locked(st, result)
                    return

                st = self._new_state(result.camera_id, result.source)
                self.by_camera[result.camera_id] = st

            if st.state in ("candidate", "confirmed") and not self._can_merge(st, result):
                self._finalize_locked(result.camera_id, force=(st.state == "confirmed"))
                st = self._new_state(result.camera_id, result.source)
                self.by_camera[result.camera_id] = st

            self._append_or_suppress_locked(st, result)
            self._advance_state_locked(st)

    def close_all(self):
        self._stop_event.set()
        try:
            self._sweeper.join(timeout=3.0)
        except Exception:
            pass

        with self.lock:
            for camera_id in list(self.by_camera.keys()):
                st = self.by_camera.get(camera_id)
                force = bool(st and st.state == "confirmed")
                self._finalize_locked(camera_id, force=force)

    def _sweeper_loop(self):
        while not self._stop_event.is_set():
            stale_items: List[tuple[str, bool]] = []

            with self.lock:
                now_wall = time.time()
                for camera_id, st in list(self.by_camera.items()):
                    if st.state not in ("candidate", "confirmed"):
                        continue

                    idle_for = now_wall - float(st.last_update_wall_ts)
                    if idle_for < self.stale_finalize_sec:
                        continue

                    if st.state == "candidate" and self._can_confirm(
                        st,
                        current_prob=st.segments[-1].result.fight_prob if st.segments else 0.0,
                        vote_enter=st.vote_count(self.keep_thr),
                    ):
                        st.state = "confirmed"
                        st.confirmed_since_ts = st.start_ts
                        st.alarm_sent = True

                    stale_items.append((camera_id, st.state == "confirmed"))

            for camera_id, force in stale_items:
                self.finalize(camera_id, force=force)

            time.sleep(self.sweep_interval_sec)

    def finalize(self, camera_id: str, force: bool = False):
        with self.lock:
            self._finalize_locked(camera_id, force=force)

    def _new_state(self, camera_id: str, source: str) -> TemporalIncidentState:
        idx = self.counter.get(camera_id, 0) + 1
        self.counter[camera_id] = idx
        incident_id = f"{camera_id}_incident_{idx:06d}"
        st = TemporalIncidentState(
            incident_id=incident_id,
            camera_id=camera_id,
            source=source,
            state="idle",
        )
        st.recent_scores = deque(maxlen=self.vote_window)
        st.recent_is_fight = deque(maxlen=self.vote_window)
        return st

    @staticmethod
    def _temporal_iou(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
        inter = max(0.0, min(float(a_end), float(b_end)) - max(float(a_start), float(b_start)))
        if inter <= 0.0:
            return 0.0
        union = max(float(a_end), float(b_end)) - min(float(a_start), float(b_start))
        if union <= 0.0:
            return 0.0
        return inter / union

    def _append_or_suppress_locked(self, st: TemporalIncidentState, result: Stage3Result) -> None:
        seg = IncidentSegment(result=result)

        if not st.segments:
            st.add_segment(seg)
            return

        last = st.segments[-1].result
        same_event = str(last.event_id) == str(result.event_id)
        tiou = self._temporal_iou(
            last.event_start_ts,
            last.event_end_ts,
            result.event_start_ts,
            result.event_end_ts,
        )

        should_suppress = same_event or (
            tiou >= self.temporal_iou_merge_thr
            and abs(float(last.event_start_ts) - float(result.event_start_ts)) <= 1.0
        )

        if not should_suppress:
            st.add_segment(seg)
            return

        st.last_update_wall_ts = time.time()

        if float(result.fight_prob) >= float(last.fight_prob):
            st.segments[-1] = seg
            try:
                st.recent_scores.pop()
            except Exception:
                pass
            try:
                st.recent_is_fight.pop()
            except Exception:
                pass
            st.recent_scores.append(float(result.fight_prob))
            st.recent_is_fight.append(1 if float(result.fight_prob) >= 0.5 else 0)
            st.last_event_end_ts = float(result.event_end_ts)
        else:
            st.last_event_end_ts = max(float(last.event_end_ts), float(result.event_end_ts))

    def _can_merge(self, st: TemporalIncidentState, result: Stage3Result) -> bool:
        if st.last_event_end_ts is None:
            return True

        gap = st.last_gap_to(result)
        if gap <= self.merge_gap_sec:
            return True

        if st.state == "confirmed" and gap <= (self.merge_gap_sec * 1.5):
            recent_nonfight = sum(1 for x in st.recent_scores if float(x) < self.keep_thr)
            return recent_nonfight <= self.max_bridge_nonfight

        return False

    def _advance_state_locked(self, st: TemporalIncidentState):
        current_prob = st.segments[-1].result.fight_prob if st.segments else 0.0
        vote_enter = st.vote_count(self.keep_thr)
        vote_keep = st.vote_count(self.keep_thr)

        if st.state == "idle":
            if (
                float(current_prob) >= self.enter_thr
                or vote_enter >= self.vote_enter_needed
                or st.max_prob >= self.single_strong_fight_thr
            ):
                st.state = "candidate"
                st.candidate_since_ts = st.start_ts

                if self._can_confirm(st, current_prob=current_prob, vote_enter=vote_enter):
                    st.state = "confirmed"
                    st.confirmed_since_ts = st.start_ts
                    st.alarm_sent = True
            return

        if st.state == "candidate":
            if self._can_confirm(st, current_prob=current_prob, vote_enter=vote_enter):
                st.state = "confirmed"
                st.confirmed_since_ts = st.start_ts
                st.alarm_sent = True
                return

            if (
                st.part_count >= self.vote_window
                and vote_keep < self.vote_keep_needed
                and float(current_prob) < self.keep_thr
                and st.max_prob < self.single_strong_fight_thr
            ):
                st.state = "idle"
                st.segments.clear()
                st.recent_scores.clear()
                st.recent_is_fight.clear()
            return

        if st.state == "confirmed":
            if float(current_prob) >= self.keep_thr:
                return
            if vote_keep >= self.vote_keep_needed:
                return
            return

    def _can_confirm(self, st: TemporalIncidentState, current_prob: float, vote_enter: int) -> bool:
        if st.part_count < self.min_incident_segments:
            if st.max_prob >= self.single_strong_fight_thr:
                return True
            return False

        if st.duration_sec < self.confirm_min_duration_sec and st.max_prob < self.single_strong_fight_thr:
            return False

        if float(current_prob) >= self.enter_thr:
            return True

        if vote_enter >= self.vote_enter_needed:
            return True

        if st.max_prob >= self.single_strong_fight_thr:
            return True

        return False

    def _finalize_locked(self, camera_id: str, force: bool = False):
        st = self.by_camera.get(camera_id)
        if st is None:
            return

        if not st.segments:
            self.by_camera.pop(camera_id, None)
            return

        final_label = self._final_label(st)

        if final_label != "fight" and not self.write_nonfight_incidents:
            self.by_camera.pop(camera_id, None)
            return

        if not force and st.state != "confirmed":
            self.by_camera.pop(camera_id, None)
            return

        if not self._wait_clips_ready(st):
            self.by_camera.pop(camera_id, None)
            return

        ts = datetime.fromtimestamp(st.start_ts).strftime("%Y%m%d_%H%M%S_%f")[:-3]
        out_path = self.out_dir / f"{st.incident_id}__{ts}__{final_label}.mp4"

        ok = self._concat_mp4s([seg.result.clip_path for seg in st.segments], out_path)
        if not ok:
            self.by_camera.pop(camera_id, None)
            return

        row = {
            "camera_id": st.camera_id,
            "source": st.source,
            "incident_id": st.incident_id,
            "start_ts": self._fmt_ts(st.start_ts),
            "end_ts": self._fmt_ts(st.end_ts),
            "start_ts_epoch": float(st.start_ts),
            "end_ts_epoch": float(st.end_ts),
            "part_count": st.part_count,
            "max_prob": round(float(st.max_prob), 6),
            "mean_prob": round(float(st.mean_prob), 6),
            "final_label": final_label,
            "clip_path": str(out_path),
            "alarm_sent": bool(st.alarm_sent),
            "parts": [
                {
                    "event_id": seg.result.event_id,
                    "fight_prob": round(float(seg.result.fight_prob), 6),
                    "fight_label": seg.result.fight_label,
                    "clip_path": seg.result.clip_path,
                    "event_start_ts": float(seg.result.event_start_ts),
                    "event_end_ts": float(seg.result.event_end_ts),
                }
                for seg in st.segments
            ],
            "created_at": self._fmt_ts(time.time()),
        }
        self._append_jsonl(self.incidents_jsonl, row)

        print(
            f"[INCIDENT] camera={st.camera_id} incident={st.incident_id} "
            f"parts={st.part_count} label={final_label} max_prob={st.max_prob:.4f} "
            f"mean_prob={st.mean_prob:.4f} out={out_path}"
        )

        if not self.keep_temp_parts:
            for seg in st.segments:
                try:
                    Path(seg.result.clip_path).unlink(missing_ok=True)
                except Exception:
                    pass

        cooldown_state = self._new_state(st.camera_id, st.source)
        cooldown_state.state = "cooldown"
        cooldown_state.cooldown_until_ts = float(st.end_ts) + self.cooldown_sec
        cooldown_state.segments = list(st.segments)
        cooldown_state.last_event_end_ts = st.end_ts
        cooldown_state.alarm_sent = True
        cooldown_state.last_update_wall_ts = time.time()
        self.by_camera[camera_id] = cooldown_state

    def _final_label(self, st: TemporalIncidentState) -> str:
        if st.state == "confirmed":
            return "fight"

        if st.max_prob >= self.single_strong_fight_thr:
            return "fight"

        if st.part_count >= self.min_incident_segments:
            if st.vote_count(self.keep_thr) >= self.vote_keep_needed:
                return "fight"
            if st.mean_prob >= max(0.50, self.keep_thr):
                return "fight"

        return "non_fight"

    def _clips_ready(self, st: TemporalIncidentState) -> bool:
        for seg in st.segments:
            clip_path = Path(seg.result.clip_path)
            if not clip_path.exists():
                return False
            try:
                if clip_path.stat().st_size <= 0:
                    return False
            except Exception:
                return False
        return True

    def _wait_clips_ready(self, st: TemporalIncidentState) -> bool:
        deadline = time.time() + self.clip_ready_wait_sec
        while time.time() < deadline:
            if self._clips_ready(st):
                return True
            time.sleep(0.15)
        return self._clips_ready(st)

    def _concat_mp4s(self, clip_paths: List[str], out_path: Path) -> bool:
        valid = [str(Path(p)) for p in clip_paths if Path(p).exists()]
        if not valid:
            return False

        if len(valid) == 1:
            try:
                src = Path(valid[0])
                if src.resolve() != out_path.resolve():
                    shutil.copy2(src, out_path)
                return out_path.exists() and out_path.stat().st_size > 0
            except Exception:
                pass

        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg:
            with tempfile.TemporaryDirectory() as td:
                lst = Path(td) / "inputs.txt"

                with open(lst, "w", encoding="utf-8") as f:
                    for p in valid:
                        safe_p = p.replace("'", r"'\''")
                        f.write(f"file '{safe_p}'\n")

                cmd_copy = [
                    ffmpeg,
                    "-y",
                    "-f",
                    "concat",
                    "-safe",
                    "0",
                    "-i",
                    str(lst),
                    "-c",
                    "copy",
                    str(out_path),
                ]
                res = subprocess.run(
                    cmd_copy,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                )
                if res.returncode == 0 and out_path.exists() and out_path.stat().st_size > 0:
                    return True

                cmd_reencode = [
                    ffmpeg,
                    "-y",
                    "-f",
                    "concat",
                    "-safe",
                    "0",
                    "-i",
                    str(lst),
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    "-an",
                    str(out_path),
                ]
                res = subprocess.run(
                    cmd_reencode,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                )
                if res.returncode == 0 and out_path.exists() and out_path.stat().st_size > 0:
                    return True

        return self._concat_with_opencv(valid, out_path)

    def _concat_with_opencv(self, clip_paths: List[str], out_path: Path) -> bool:
        writer = None
        try:
            target_size = None
            target_fps = None

            for p in clip_paths:
                cap = cv2.VideoCapture(p)
                if not cap.isOpened():
                    continue

                fps = cap.get(cv2.CAP_PROP_FPS)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                if width > 0 and height > 0 and target_size is None:
                    target_size = (width, height)
                if fps and fps > 1 and target_fps is None:
                    target_fps = float(fps)

                cap.release()

                if target_size is not None and target_fps is not None:
                    break

            if target_size is None:
                return False
            if target_fps is None:
                target_fps = 16.0

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(out_path), fourcc, float(target_fps), target_size)
            if not writer.isOpened():
                return False

            for p in clip_paths:
                cap = cv2.VideoCapture(p)
                if not cap.isOpened():
                    continue

                while True:
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        break

                    if (frame.shape[1], frame.shape[0]) != target_size:
                        frame = cv2.resize(frame, target_size)

                    writer.write(frame)

                cap.release()

            writer.release()
            writer = None

            return out_path.exists() and out_path.stat().st_size > 0
        except Exception:
            try:
                if writer is not None:
                    writer.release()
            except Exception:
                pass
            return False

    @staticmethod
    def _append_jsonl(path: Path, row: dict):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    @staticmethod
    def _fmt_ts(ts: float) -> str:
        return datetime.fromtimestamp(float(ts)).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]