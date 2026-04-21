import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from django.conf import settings


@dataclass
class ActiveRun:
    process: subprocess.Popen
    run_name: str
    run_dir: Path
    sources: list[dict]
    stdout_path: Path
    stderr_path: Path
    started_at: float


def _append_if_present(cmd: list[str], defaults: dict, key: str, flag: str):
    value = defaults.get(key, None)
    if value is None:
        return
    cmd.extend([flag, str(value)])


def build_command(sources: list[dict], run_name: str) -> list[str]:
    defaults = settings.PIPELINE_DEFAULTS

    cmd = [
        sys.executable,
        "-m",
        settings.PIPELINE_ENTRY_MODULE,
    ]

    for item in sources:
        cmd.extend(["--camera", f"{item['camera_id']}={item['source']}"])

    cmd.extend(["--motion-config", defaults["motion_config"]])
    cmd.extend(["--yolo-config", defaults["yolo_config"]])
    cmd.extend(["--yolo-weights", defaults["yolo_weights"]])
    cmd.extend(["--pose-config", defaults["pose_config"]])
    cmd.extend(["--stage3-config", defaults["stage3_config"]])
    cmd.extend(["--person-conf", str(defaults["person_conf"])])
    cmd.extend(["--yolo-stride", str(defaults["yolo_stride"])])
    cmd.extend(["--pose-stride", str(defaults["pose_stride"])])
    cmd.extend(["--fight-thr", str(defaults["fight_thr"])])
    cmd.extend(["--output-dir", str(settings.PIPELINE_OUTPUT_BASE)])
    cmd.extend(["--run-name", run_name])

    _append_if_present(cmd, defaults, "roi_size", "--roi-size")
    _append_if_present(cmd, defaults, "min_persons_for_pose", "--min-persons-for-pose")
    _append_if_present(cmd, defaults, "pose_hold_frames", "--pose-hold-frames")
    _append_if_present(cmd, defaults, "event_close_grace_frames", "--event-close-grace-frames")
    _append_if_present(cmd, defaults, "prebuffer_frames", "--prebuffer-frames")
    _append_if_present(cmd, defaults, "max_event_frames", "--max-event-frames")
    _append_if_present(cmd, defaults, "clip_fps", "--clip-fps")
    _append_if_present(cmd, defaults, "reconnect_sec", "--reconnect-sec")
    _append_if_present(cmd, defaults, "status_log_every", "--status-log-every")
    _append_if_present(cmd, defaults, "min_queue_frames", "--min-queue-frames")
    _append_if_present(cmd, defaults, "stage3_queue_size", "--stage3-queue-size")

    _append_if_present(cmd, defaults, "preview_every_frames", "--preview-every-frames")
    _append_if_present(cmd, defaults, "preview_write_interval_sec", "--preview-write-interval-sec")
    _append_if_present(cmd, defaults, "preview_jpeg_quality", "--preview-jpeg-quality")
    _append_if_present(cmd, defaults, "clip_writer_queue_size", "--clip-writer-queue-size")
    _append_if_present(cmd, defaults, "report_flush_interval_sec", "--report-flush-interval-sec")
    _append_if_present(cmd, defaults, "cv2_threads", "--cv2-threads")

    if defaults.get("use_pose"):
        cmd.append("--use-pose")

    if defaults.get("use_stage3"):
        cmd.append("--use-stage3")

    return cmd


def start_pipeline(sources: list[dict]) -> ActiveRun:
    run_name = time.strftime("ui_run_%Y%m%d_%H%M%S")
    run_dir = Path(settings.PIPELINE_OUTPUT_BASE) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "previews").mkdir(parents=True, exist_ok=True)

    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"

    cmd = build_command(sources, run_name)

    env = os.environ.copy()
    repo_root = str(settings.REPO_ROOT)

    existing_pythonpath = env.get("PYTHONPATH", "").strip()
    if existing_pythonpath:
        env["PYTHONPATH"] = repo_root + os.pathsep + existing_pythonpath
    else:
        env["PYTHONPATH"] = repo_root

    stdout_handle = open(stdout_path, "a", encoding="utf-8")
    stderr_handle = open(stderr_path, "a", encoding="utf-8")

    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP

    process = subprocess.Popen(
        cmd,
        cwd=repo_root,
        env=env,
        stdout=stdout_handle,
        stderr=stderr_handle,
        text=True,
        creationflags=creationflags,
    )

    return ActiveRun(
        process=process,
        run_name=run_name,
        run_dir=run_dir,
        sources=sources,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        started_at=time.time(),
    )


def stop_pipeline(active: ActiveRun | None) -> None:
    if active is None:
        return

    process = active.process
    if process.poll() is not None:
        return

    try:
        if os.name == "nt":
            process.send_signal(signal.CTRL_BREAK_EVENT)
            process.wait(timeout=5)
        else:
            process.terminate()
            process.wait(timeout=5)
    except Exception:
        try:
            process.kill()
        except Exception:
            pass