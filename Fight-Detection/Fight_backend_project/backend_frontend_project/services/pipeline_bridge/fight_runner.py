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

    if isinstance(value, bool):
        if value:
            cmd.append(flag)
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

    # Stage3 event quality filter
    # Bunlar olmazsa settings.py içine yazdığın kalite filtresi pipeline'a gitmez.
    _append_if_present(
        cmd,
        defaults,
        "stage3_event_min_positive_hits",
        "--stage3-event-min-positive-hits",
    )
    _append_if_present(
        cmd,
        defaults,
        "stage3_event_min_pose_mean",
        "--stage3-event-min-pose-mean",
    )
    _append_if_present(
        cmd,
        defaults,
        "stage3_event_min_pose_max",
        "--stage3-event-min-pose-max",
    )
    _append_if_present(
        cmd,
        defaults,
        "stage3_event_min_duration_sec",
        "--stage3-event-min-duration-sec",
    )
    _append_if_present(
        cmd,
        defaults,
        "stage3_drop_close_reasons",
        "--stage3-drop-close-reasons",
    )

    _append_if_present(cmd, defaults, "incident_enter_thr", "--incident-enter-thr")
    _append_if_present(cmd, defaults, "incident_keep_thr", "--incident-keep-thr")
    _append_if_present(cmd, defaults, "incident_vote_window", "--incident-vote-window")
    _append_if_present(cmd, defaults, "incident_vote_enter_needed", "--incident-vote-enter-needed")
    _append_if_present(cmd, defaults, "incident_vote_keep_needed", "--incident-vote-keep-needed")
    _append_if_present(cmd, defaults, "incident_merge_gap_sec", "--incident-merge-gap-sec")
    _append_if_present(cmd, defaults, "incident_max_bridge_nonfight", "--incident-max-bridge-nonfight")
    _append_if_present(cmd, defaults, "incident_min_segments", "--incident-min-segments")
    _append_if_present(
        cmd,
        defaults,
        "incident_single_strong_fight_thr",
        "--incident-single-strong-fight-thr",
    )
    _append_if_present(
        cmd,
        defaults,
        "incident_confirm_min_duration_sec",
        "--incident-confirm-min-duration-sec",
    )
    _append_if_present(cmd, defaults, "incident_cooldown_sec", "--incident-cooldown-sec")
    _append_if_present(
        cmd,
        defaults,
        "incident_clip_ready_wait_sec",
        "--incident-clip-ready-wait-sec",
    )
    _append_if_present(
        cmd,
        defaults,
        "incident_stale_finalize_sec",
        "--incident-stale-finalize-sec",
    )
    _append_if_present(
        cmd,
        defaults,
        "incident_temporal_iou_merge_thr",
        "--incident-temporal-iou-merge-thr",
    )
    _append_if_present(cmd, defaults, "incident_write_nonfight", "--incident-write-nonfight")

    _append_if_present(cmd, defaults, "preview_every_frames", "--preview-every-frames")
    _append_if_present(
        cmd,
        defaults,
        "preview_write_interval_sec",
        "--preview-write-interval-sec",
    )
    _append_if_present(cmd, defaults, "preview_jpeg_quality", "--preview-jpeg-quality")
    _append_if_present(cmd, defaults, "clip_writer_queue_size", "--clip-writer-queue-size")
    _append_if_present(
        cmd,
        defaults,
        "report_flush_interval_sec",
        "--report-flush-interval-sec",
    )
    _append_if_present(cmd, defaults, "cv2_threads", "--cv2-threads")

    if defaults.get("use_pose"):
        cmd.append("--use-pose")

    if defaults.get("use_stage3"):
        cmd.append("--use-stage3")

    return cmd


def _tail_file(path: Path, max_chars: int = 6000) -> str:
    try:
        if not path.exists():
            return ""
        text = path.read_text(encoding="utf-8", errors="replace")
        return text[-max_chars:]
    except Exception as exc:
        return f"<log okunamadı: {exc}>"


def _write_command_debug(run_dir: Path, cmd: list[str], env: dict):
    try:
        lines = []
        lines.append("COMMAND:")
        lines.append(" ".join(f'"{x}"' if " " in str(x) else str(x) for x in cmd))
        lines.append("")
        lines.append(f"CWD: {settings.REPO_ROOT}")
        lines.append(f"PYTHON: {sys.executable}")
        lines.append(f"PYTHONPATH: {env.get('PYTHONPATH', '')}")
        lines.append("")
        lines.append("ARGS:")
        for x in cmd:
            lines.append(str(x))

        (run_dir / "command.txt").write_text("\n".join(lines), encoding="utf-8")
    except Exception:
        pass


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

    _write_command_debug(run_dir, cmd, env)

    stdout_handle = open(stdout_path, "a", encoding="utf-8", buffering=1)
    stderr_handle = open(stderr_path, "a", encoding="utf-8", buffering=1)

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

    # Model init / argüman hatası varsa process hemen kapanır.
    # UI bunu sessiz "başladı sonra kapandı" gibi göstermesin.
    time.sleep(2.0)
    rc = process.poll()

    if rc is not None:
        try:
            stdout_handle.close()
        except Exception:
            pass
        try:
            stderr_handle.close()
        except Exception:
            pass

        stdout_tail = _tail_file(stdout_path)
        stderr_tail = _tail_file(stderr_path)
        cmd_text = " ".join(cmd)

        raise RuntimeError(
            "Pipeline başladıktan hemen sonra kapandı.\n\n"
            f"return_code={rc}\n"
            f"run_dir={run_dir}\n\n"
            f"COMMAND:\n{cmd_text}\n\n"
            f"STDOUT tail:\n{stdout_tail}\n\n"
            f"STDERR tail:\n{stderr_tail}\n"
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