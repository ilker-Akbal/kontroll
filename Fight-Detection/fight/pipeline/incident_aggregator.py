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

    @property
    def start_ts(self) -> float:
        return float(self.result.event_start_ts)

    @property
    def end_ts(self) -> float:
        return float(self.result.event_end_ts)

    @property
    def prob(self) -> float:
        return float(self.result.fight_prob)


@dataclass
class TemporalIncidentState:
    incident_id: str
    camera_id: str
    source: str

    # idle -> candidate -> confirmed -> cooldown
    state: str = "idle"

    segments: List[IncidentSegment] = field(default_factory=list)
    recent_scores: Deque[float] = field(default_factory=lambda: deque(maxlen=7))

    candidate_since_ts: Optional[float] = None
    confirmed_since_ts: Optional[float] = None

    last_event_end_ts: Optional[float] = None
    last_update_wall_ts: float = field(default_factory=time.time)

    alarm_sent: bool = False
    cooldown_until_ts: float = 0.0

    def add_segment(self, seg: IncidentSegment, vote_window: int) -> None:
        self.segments.append(seg)
        self.recent_scores = deque(self.recent_scores, maxlen=int(vote_window))
        self.recent_scores.append(float(seg.prob))
        self.last_event_end_ts = float(seg.end_ts)
        self.last_update_wall_ts = time.time()

    @property
    def start_ts(self) -> float:
        if not self.segments:
            return 0.0
        return min(s.start_ts for s in self.segments)

    @property
    def end_ts(self) -> float:
        if not self.segments:
            return 0.0
        return max(s.end_ts for s in self.segments)

    @property
    def duration_sec(self) -> float:
        return max(0.0, self.end_ts - self.start_ts)

    @property
    def part_count(self) -> int:
        return len(self.segments)

    @property
    def max_prob(self) -> float:
        if not self.segments:
            return 0.0
        return max(s.prob for s in self.segments)

    @property
    def mean_prob(self) -> float:
        if not self.segments:
            return 0.0
        vals = [s.prob for s in self.segments]
        return sum(vals) / max(1, len(vals))

    @property
    def topk_mean_prob(self) -> float:
        if not self.segments:
            return 0.0
        vals = sorted([s.prob for s in self.segments], reverse=True)
        vals = vals[: min(3, len(vals))]
        return sum(vals) / max(1, len(vals))

    @property
    def decision_score(self) -> float:
        """
        Final karar skoru.
        Sadece mean kullanırsak uzun olaylarda düşük clipler skoru gereksiz düşürür.
        Sadece max kullanırsak tek yanlış pozitif alarm yapabilir.
        Bu yüzden max + top-k mean karışımı kullanıyoruz.
        """
        return (0.65 * self.max_prob) + (0.35 * self.topk_mean_prob)

    def vote_count(self, thr: float) -> int:
        return sum(1 for x in self.recent_scores if float(x) >= float(thr))

    def last_gap_to(self, result: Stage3Result) -> float:
        if self.last_event_end_ts is None:
            return 0.0
        return float(result.event_start_ts) - float(self.last_event_end_ts)


class IncidentAggregator:
    """
    Son karar katmanı.

    Görevi:
    - Stage3 clip sonuçlarını kamera bazında zamansal olarak toplar.
    - Birbirine çok benzeyen / overlap olan clipleri Temporal NMS ile bastırır.
    - Yakın zamanlı clipleri aynı incident altında birleştirir.
    - Tek cliplik çok güçlü kavga skorunu kaçırmaz.
    - Zayıf tekil false-positive skorları alarm yapmaz.
    - Cooldown ile aynı kavga için sürekli alarm üretmez.

    Önemli çıktı:
    - output_dir/incidents.jsonl
    - output_dir/incidents/*.mp4
    """

    def __init__(
        self,
        out_dir: str,
        merge_gap_sec: float = 10.0,
        max_bridge_nonfight: int = 2,
        enter_thr: float = 0.62,
        keep_thr: float = 0.50,
        vote_window: int = 7,
        vote_enter_needed: int = 2,
        vote_keep_needed: int = 2,
        min_incident_segments: int = 2,
        single_strong_fight_thr: float = 0.82,
        confirm_min_duration_sec: float = 0.8,
        cooldown_sec: float = 18.0,
        keep_temp_parts: bool = True,
        write_nonfight_incidents: bool = False,
        clip_ready_wait_sec: float = 8.0,
        stale_finalize_sec: float = 4.0,
        temporal_iou_merge_thr: float = 0.42,
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

        self.lock = threading.RLock()
        self.by_camera: Dict[str, TemporalIncidentState] = {}
        self.counter: Dict[str, int] = {}

        self._stop_event = threading.Event()
        self._sweeper = threading.Thread(
            target=self._sweeper_loop,
            name="incident_sweeper",
            daemon=True,
        )
        self._sweeper.start()

    def submit(self, result: Stage3Result) -> None:
        """
        Stage3Worker burayı çağırır.
        Burada direkt alarm yok; karar Temporal NMS + aggregation sonrası verilir.
        """
        with self.lock:
            result = self._normalize_result(result)

            st = self.by_camera.get(result.camera_id)
            if st is None:
                st = self._new_state(result.camera_id, result.source)
                self.by_camera[result.camera_id] = st

            # Cooldown içindeyken gelen yeni clip aynı olayla ilişkiliyse mevcut state'e eklenebilir
            # ama yeni alarm üretilmez.
            if st.state == "cooldown":
                if float(result.event_start_ts) <= float(st.cooldown_until_ts):
                    if self._can_merge(st, result):
                        self._append_or_suppress_locked(st, result)
                    return

                st = self._new_state(result.camera_id, result.source)
                self.by_camera[result.camera_id] = st

            # Zaman olarak çok uzaksa eski incident finalize edilir.
            if st.state in ("candidate", "confirmed") and not self._can_merge(st, result):
                self._finalize_locked(result.camera_id, force=(st.state == "confirmed"))
                st = self._new_state(result.camera_id, result.source)
                self.by_camera[result.camera_id] = st

            self._append_or_suppress_locked(st, result)
            self._advance_state_locked(st)

    def close_all(self) -> None:
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

    def finalize(self, camera_id: str, force: bool = False) -> None:
        with self.lock:
            self._finalize_locked(camera_id, force=force)

    def _sweeper_loop(self) -> None:
        while not self._stop_event.is_set():
            finalize_items: List[tuple[str, bool]] = []

            with self.lock:
                now_wall = time.time()

                for camera_id, st in list(self.by_camera.items()):
                    if st.state not in ("candidate", "confirmed"):
                        continue

                    idle_for = now_wall - float(st.last_update_wall_ts)
                    if idle_for < self.stale_finalize_sec:
                        continue

                    if st.state == "candidate":
                        if self._can_confirm(st):
                            st.state = "confirmed"
                            st.confirmed_since_ts = st.start_ts
                            st.alarm_sent = True

                    finalize_items.append((camera_id, st.state == "confirmed"))

            for camera_id, force in finalize_items:
                self.finalize(camera_id, force=force)

            time.sleep(self.sweep_interval_sec)

    def _normalize_result(self, result: Stage3Result) -> Stage3Result:
        start_ts = float(result.event_start_ts)
        end_ts = float(result.event_end_ts)
        if end_ts < start_ts:
            start_ts, end_ts = end_ts, start_ts

        prob = max(0.0, min(1.0, float(result.fight_prob)))

        return Stage3Result(
            camera_id=str(result.camera_id),
            source=str(result.source),
            event_id=str(result.event_id),
            event_start_ts=start_ts,
            event_end_ts=end_ts,
            clip_path=str(result.clip_path),
            fight_prob=prob,
            fight_label=str(result.fight_label),
            pose_score_max=float(result.pose_score_max),
            pose_score_mean=float(result.pose_score_mean),
        )

    def _new_state(self, camera_id: str, source: str) -> TemporalIncidentState:
        idx = self.counter.get(camera_id, 0) + 1
        self.counter[camera_id] = idx

        st = TemporalIncidentState(
            incident_id=f"{camera_id}_incident_{idx:06d}",
            camera_id=str(camera_id),
            source=str(source),
            state="idle",
        )
        st.recent_scores = deque(maxlen=self.vote_window)
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

    def _temporal_overlap_ratio(self, a: Stage3Result, b: Stage3Result) -> float:
        """
        Kısa clip uzun clip içinde kalırsa IoU düşük çıkabilir.
        Bu yüzden overlap / kısa_segment de bakıyoruz.
        """
        a_len = max(1e-6, float(a.event_end_ts) - float(a.event_start_ts))
        b_len = max(1e-6, float(b.event_end_ts) - float(b.event_start_ts))

        inter = max(
            0.0,
            min(float(a.event_end_ts), float(b.event_end_ts))
            - max(float(a.event_start_ts), float(b.event_start_ts)),
        )
        return inter / max(1e-6, min(a_len, b_len))

    def _append_or_suppress_locked(self, st: TemporalIncidentState, result: Stage3Result) -> None:
        """
        Temporal NMS:
        Aynı event_id veya ciddi overlap varsa düşük skorlu segmenti bastır,
        yüksek skorlu olanı sakla.
        """
        new_seg = IncidentSegment(result=result)

        if not st.segments:
            st.add_segment(new_seg, self.vote_window)
            return

        suppress_idx = None
        best_overlap = 0.0

        for i in range(len(st.segments) - 1, -1, -1):
            old = st.segments[i].result

            same_event = str(old.event_id) == str(result.event_id)

            tiou = self._temporal_iou(
                old.event_start_ts,
                old.event_end_ts,
                result.event_start_ts,
                result.event_end_ts,
            )
            cover = self._temporal_overlap_ratio(old, result)

            # Son birkaç saniyedeki overlap segmentler aynı olayın farklı kırpımıdır.
            close_start = abs(float(old.event_start_ts) - float(result.event_start_ts)) <= self.merge_gap_sec

            should_suppress = same_event or (
                close_start and (tiou >= self.temporal_iou_merge_thr or cover >= 0.70)
            )

            if should_suppress and max(tiou, cover) >= best_overlap:
                best_overlap = max(tiou, cover)
                suppress_idx = i

        if suppress_idx is None:
            st.add_segment(new_seg, self.vote_window)
            return

        old_seg = st.segments[suppress_idx]

        # Yüksek skorlu segmenti tut. Böylece Stage3 tekrar tekrar aynı olayı işlerse
        # tek alarm/tek incident olur.
        if new_seg.prob >= old_seg.prob:
            st.segments[suppress_idx] = new_seg

        st.last_update_wall_ts = time.time()
        st.last_event_end_ts = max(
            float(st.last_event_end_ts or result.event_end_ts),
            float(result.event_end_ts),
        )

        # NMS sonrası vote history'yi güncel segmentlerden yeniden kur.
        self._rebuild_recent_scores(st)

    def _rebuild_recent_scores(self, st: TemporalIncidentState) -> None:
        vals = [seg.prob for seg in st.segments[-self.vote_window :]]
        st.recent_scores = deque(vals, maxlen=self.vote_window)

    def _can_merge(self, st: TemporalIncidentState, result: Stage3Result) -> bool:
        if not st.segments or st.last_event_end_ts is None:
            return True

        gap = st.last_gap_to(result)

        # Overlap veya temas varsa kesin aynı olay.
        if gap <= self.merge_gap_sec:
            return True

        # Çok güçlü bir confirmed olaydan sonra kısa kopmalar olabilir.
        # Araya 1-2 tane düşük skor girse bile tek incident tut.
        if st.state == "confirmed" and gap <= (self.merge_gap_sec * 1.5):
            recent_nonfight = sum(1 for x in st.recent_scores if float(x) < self.keep_thr)
            return recent_nonfight <= self.max_bridge_nonfight

        return False

    def _advance_state_locked(self, st: TemporalIncidentState) -> None:
        if not st.segments:
            st.state = "idle"
            return

        current_prob = st.segments[-1].prob
        vote_enter = st.vote_count(self.keep_thr)

        if st.state == "idle":
            if (
                current_prob >= self.enter_thr
                or st.max_prob >= self.single_strong_fight_thr
                or vote_enter >= self.vote_enter_needed
            ):
                st.state = "candidate"
                st.candidate_since_ts = st.start_ts

                if self._can_confirm(st):
                    st.state = "confirmed"
                    st.confirmed_since_ts = st.start_ts
                    st.alarm_sent = True
            return

        if st.state == "candidate":
            if self._can_confirm(st):
                st.state = "confirmed"
                st.confirmed_since_ts = st.start_ts
                st.alarm_sent = True
                return

            # Eski sürümde burada segmentler tamamen silinebiliyordu.
            # Bu, "Stage3 fight dedi ama alarm yok" hissini artırır.
            # Artık sadece çok zayıf ve yeterince beklemiş adayları düşürüyoruz.
            if (
                st.part_count >= self.vote_window
                and st.max_prob < self.enter_thr
                and st.decision_score < self.keep_thr
                and st.vote_count(self.keep_thr) == 0
            ):
                st.state = "idle"
                st.segments.clear()
                st.recent_scores.clear()
            return

        if st.state == "confirmed":
            # Confirmed olduktan sonra final kararı sweeper/finalize verir.
            return

    def _can_confirm(self, st: TemporalIncidentState) -> bool:
        if not st.segments:
            return False

        current_prob = st.segments[-1].prob
        vote_enter = st.vote_count(self.keep_thr)

        # Çok güçlü tek clip: kaçırma.
        if st.max_prob >= self.single_strong_fight_thr:
            return True

        # En az 2 segmentli normal doğrulama.
        if st.part_count < self.min_incident_segments:
            return False

        # Süre çok kısaysa ve çok güçlü skor yoksa alarm verme.
        if st.duration_sec < self.confirm_min_duration_sec:
            return False

        if current_prob >= self.enter_thr and vote_enter >= max(1, self.vote_enter_needed - 1):
            return True

        if vote_enter >= self.vote_enter_needed:
            return True

        if st.decision_score >= self.enter_thr and st.vote_count(self.keep_thr) >= self.vote_keep_needed:
            return True

        return False

    def _finalize_locked(self, camera_id: str, force: bool = False) -> None:
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
            print(
                f"[INCIDENT][WARN] clips not ready, skip incident camera={st.camera_id} "
                f"incident={st.incident_id}"
            )
            self.by_camera.pop(camera_id, None)
            return

        ts = datetime.fromtimestamp(st.start_ts).strftime("%Y%m%d_%H%M%S_%f")[:-3]
        out_path = self.out_dir / f"{st.incident_id}__{ts}__{final_label}.mp4"

        ok = self._concat_mp4s([seg.result.clip_path for seg in st.segments], out_path)
        if not ok:
            print(
                f"[INCIDENT][WARN] concat failed camera={st.camera_id} "
                f"incident={st.incident_id}"
            )
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
            "duration_sec": round(float(st.duration_sec), 3),

            "part_count": int(st.part_count),
            "max_prob": round(float(st.max_prob), 6),
            "mean_prob": round(float(st.mean_prob), 6),
            "topk_mean_prob": round(float(st.topk_mean_prob), 6),
            "decision_score": round(float(st.decision_score), 6),

            "enter_thr": float(self.enter_thr),
            "keep_thr": float(self.keep_thr),
            "single_strong_fight_thr": float(self.single_strong_fight_thr),

            "final_label": final_label,
            "clip_path": str(out_path),
            "alarm_sent": bool(final_label == "fight"),
            "state": st.state,

            "parts": [
                {
                    "event_id": seg.result.event_id,
                    "fight_prob": round(float(seg.result.fight_prob), 6),
                    "fight_label": seg.result.fight_label,
                    "pose_score_max": round(float(seg.result.pose_score_max), 6),
                    "pose_score_mean": round(float(seg.result.pose_score_mean), 6),
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
            f"parts={st.part_count} label={final_label} "
            f"max={st.max_prob:.4f} mean={st.mean_prob:.4f} "
            f"decision={st.decision_score:.4f} out={out_path}"
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
        cooldown_state.recent_scores = deque(
            [seg.prob for seg in st.segments[-self.vote_window :]],
            maxlen=self.vote_window,
        )

        self.by_camera[camera_id] = cooldown_state

    def _final_label(self, st: TemporalIncidentState) -> str:
        if st.max_prob >= self.single_strong_fight_thr:
            return "fight"

        if st.state == "confirmed":
            if st.decision_score >= self.keep_thr:
                return "fight"
            if st.vote_count(self.keep_thr) >= self.vote_keep_needed:
                return "fight"

        if st.part_count >= self.min_incident_segments:
            if st.decision_score >= self.enter_thr:
                return "fight"
            if st.vote_count(self.keep_thr) >= self.vote_enter_needed:
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

        out_path.parent.mkdir(parents=True, exist_ok=True)

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
                        safe_p = str(Path(p).resolve()).replace("'", r"'\''")
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

            wrote_any = False

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
                    wrote_any = True

                cap.release()

            writer.release()
            writer = None

            return wrote_any and out_path.exists() and out_path.stat().st_size > 0

        except Exception:
            try:
                if writer is not None:
                    writer.release()
            except Exception:
                pass
            return False

    @staticmethod
    def _append_jsonl(path: Path, row: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    @staticmethod
    def _fmt_ts(ts: float) -> str:
        return datetime.fromtimestamp(float(ts)).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]