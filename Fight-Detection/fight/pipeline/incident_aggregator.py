from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List


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
class IncidentState:
    incident_id: str
    camera_id: str
    source: str
    start_ts: float
    end_ts: float
    parts: List[Stage3Result] = field(default_factory=list)
    last_update_ts: float = field(default_factory=time.time)

    def add(self, result: Stage3Result):
        self.parts.append(result)
        self.end_ts = max(self.end_ts, result.event_end_ts)
        self.last_update_ts = time.time()

    @property
    def max_prob(self) -> float:
        return max((p.fight_prob for p in self.parts), default=0.0)

    @property
    def mean_prob(self) -> float:
        vals = [p.fight_prob for p in self.parts]
        return sum(vals) / max(1, len(vals))

    @property
    def strong_count(self) -> int:
        return sum(1 for p in self.parts if p.fight_prob >= 0.60)

    @property
    def medium_count(self) -> int:
        return sum(1 for p in self.parts if p.fight_prob >= 0.52)

    @property
    def weak_negative_count(self) -> int:
        return sum(1 for p in self.parts if p.fight_prob < 0.45)

    def final_label(self) -> str:
        # Daha agresif birleştirme:
        # arada 1 tane non_fight gelse bile çoğunluk fight ise olayı fight say
        if self.max_prob >= 0.70:
            return "fight"
        if self.strong_count >= 2:
            return "fight"
        if len(self.parts) >= 3 and self.medium_count >= 2 and self.weak_negative_count <= 1:
            return "fight"
        if len(self.parts) >= 4 and self.medium_count >= 3:
            return "fight"
        if self.mean_prob >= 0.58 and self.medium_count >= 2:
            return "fight"
        return "non_fight"


class IncidentAggregator:
    def __init__(
        self,
        out_dir: str,
        merge_gap_sec: float = 8.0,
        finalize_idle_sec: float = 6.0,
        keep_temp_parts: bool = True,
        write_nonfight_incidents: bool = False,
    ):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.incidents_jsonl = self.out_dir.parent / "incidents.jsonl"

        self.merge_gap_sec = float(merge_gap_sec)
        self.finalize_idle_sec = float(finalize_idle_sec)
        self.keep_temp_parts = bool(keep_temp_parts)
        self.write_nonfight_incidents = bool(write_nonfight_incidents)

        self.lock = threading.Lock()
        self.active: Dict[str, IncidentState] = {}
        self.counter: Dict[str, int] = {}

    def submit(self, result: Stage3Result):
        with self.lock:
            st = self.active.get(result.camera_id)

            if st is None:
                st = self._new_incident(result)
                self.active[result.camera_id] = st
                return

            gap = float(result.event_start_ts) - float(st.end_ts)

            if gap <= self.merge_gap_sec:
                st.add(result)
            else:
                self._finalize_locked(result.camera_id)
                st = self._new_incident(result)
                self.active[result.camera_id] = st

    def flush_expired(self):
        now_ts = time.time()
        with self.lock:
            to_finalize = []
            for camera_id, st in self.active.items():
                if (now_ts - st.last_update_ts) >= self.finalize_idle_sec:
                    to_finalize.append(camera_id)

            for camera_id in to_finalize:
                self._finalize_locked(camera_id)

    def close_all(self):
        with self.lock:
            for camera_id in list(self.active.keys()):
                self._finalize_locked(camera_id)

    def _new_incident(self, result: Stage3Result) -> IncidentState:
        idx = self.counter.get(result.camera_id, 0) + 1
        self.counter[result.camera_id] = idx

        incident_id = f"{result.camera_id}_incident_{idx:06d}"
        st = IncidentState(
            incident_id=incident_id,
            camera_id=result.camera_id,
            source=result.source,
            start_ts=result.event_start_ts,
            end_ts=result.event_end_ts,
        )
        st.add(result)
        return st

    def _finalize_locked(self, camera_id: str):
        st = self.active.pop(camera_id, None)
        if st is None or not st.parts:
            return

        final_label = st.final_label()

        # non_fight incident'leri kullanıcıya kayıt olarak verme
        if final_label != "fight" and not self.write_nonfight_incidents:
            return

        ts = datetime.fromtimestamp(st.start_ts).strftime("%Y%m%d_%H%M%S_%f")[:-3]
        out_path = self.out_dir / f"{st.incident_id}__{ts}__{final_label}.mp4"

        ok = self._concat_mp4s([p.clip_path for p in st.parts], out_path)
        if not ok:
            return

        row = {
            "camera_id": st.camera_id,
            "source": st.source,
            "incident_id": st.incident_id,
            "start_ts": self._fmt_ts(st.start_ts),
            "end_ts": self._fmt_ts(st.end_ts),
            "start_ts_epoch": float(st.start_ts),
            "end_ts_epoch": float(st.end_ts),
            "part_count": len(st.parts),
            "max_prob": round(float(st.max_prob), 6),
            "mean_prob": round(float(st.mean_prob), 6),
            "final_label": final_label,
            "clip_path": str(out_path),
            "parts": [
                {
                    "event_id": p.event_id,
                    "fight_prob": round(float(p.fight_prob), 6),
                    "fight_label": p.fight_label,
                    "clip_path": p.clip_path,
                    "event_start_ts": float(p.event_start_ts),
                    "event_end_ts": float(p.event_end_ts),
                }
                for p in st.parts
            ],
            "created_at": self._fmt_ts(time.time()),
        }
        self._append_jsonl(self.incidents_jsonl, row)

        print(
            f"[INCIDENT] camera={st.camera_id} incident={st.incident_id} "
            f"parts={len(st.parts)} label={final_label} max_prob={st.max_prob:.4f} "
            f"mean_prob={st.mean_prob:.4f} out={out_path}"
        )

        if not self.keep_temp_parts:
            for p in st.parts:
                try:
                    Path(p.clip_path).unlink(missing_ok=True)
                except Exception:
                    pass

    def _concat_mp4s(self, clip_paths: List[str], out_path: Path) -> bool:
        valid = [str(Path(p)) for p in clip_paths if Path(p).exists()]
        if not valid:
            return False

        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            return False

        with tempfile.TemporaryDirectory() as td:
            lst = Path(td) / "inputs.txt"

            with open(lst, "w", encoding="utf-8") as f:
                for p in valid:
                    safe_p = p.replace("'", r"'\''")
                    f.write(f"file '{safe_p}'\n")

            # önce copy concat dene
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

            # copy concat olmazsa re-encode fallback
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

        return False

    @staticmethod
    def _append_jsonl(path: Path, row: dict):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    @staticmethod
    def _fmt_ts(ts: float) -> str:
        return datetime.fromtimestamp(float(ts)).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]