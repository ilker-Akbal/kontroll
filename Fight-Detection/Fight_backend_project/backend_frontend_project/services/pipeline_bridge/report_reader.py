import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []

    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def _safe_float(value, default=0.0):
    try:
        if value in ("", None):
            return default
        return float(value)
    except Exception:
        return default


def _safe_int(value, default=0):
    try:
        if value in ("", None):
            return default
        return int(value)
    except Exception:
        return default


def build_dashboard_report(
    run_dir: Path,
    sources: list[dict],
    process_alive: bool,
    pid: int | None,
    started_at: float | None,
    return_code: int | None,
):
    run_dir = Path(run_dir)

    status_rows = read_jsonl(run_dir / "camera_status.jsonl")
    event_rows = read_jsonl(run_dir / "events.jsonl")
    stage3_rows = read_jsonl(run_dir / "stage3_results.jsonl")

    latest_status = {}
    for row in status_rows:
        camera_id = row.get("camera_id")
        if camera_id:
            latest_status[camera_id] = row

    latest_event = {}
    for row in event_rows:
        camera_id = row.get("camera_id")
        if camera_id:
            latest_event[camera_id] = row

    latest_stage3 = {}
    for row in stage3_rows:
        camera_id = row.get("camera_id")
        if camera_id:
            latest_stage3[camera_id] = row

    cameras = []
    for item in sources:
        camera_id = item["camera_id"]
        source = item["source"]

        status = latest_status.get(camera_id, {})
        event = latest_event.get(camera_id, {})
        stage3 = latest_stage3.get(camera_id, {})

        latest_event_id = (
            status.get("event_id")
            or event.get("event_id")
            or ""
        )

        latest_event_status = (
            status.get("latest_event_status")
            or event.get("status")
            or ""
        )

        latest_clip_path = (
            status.get("clip_path")
            or event.get("clip_path")
            or ""
        )

        latest_stage3_label = (
            status.get("latest_stage3_label")
            or stage3.get("fight_label")
            or ""
        )

        latest_stage3_prob = status.get("latest_stage3_prob")
        if latest_stage3_prob in ("", None):
            latest_stage3_prob = stage3.get("fight_prob", 0.0)

        queue_status = status.get("queue_status", "")
        queue_reason = status.get("queue_reason", "")
        queue_size = status.get("queue_size", "")
        queue_capacity = status.get("queue_capacity", "")

        cameras.append(
            {
                "camera_id": camera_id,
                "source": source,
                "stage": status.get("stage", ""),
                "detail": status.get("detail", "beklemede"),
                "last_ts": status.get("ts", ""),
                "motion_score": _safe_float(status.get("motion_score", ""), default=0.0),
                "persons": _safe_int(status.get("persons", ""), default=0),
                "pair_ok": _safe_int(status.get("pair_ok", ""), default=0),
                "pose_positive": _safe_int(status.get("pose_positive", ""), default=0),
                "pose_score": _safe_float(status.get("pose_score", ""), default=0.0),
                "event_active": _safe_int(status.get("event_active", ""), default=0),
                "latest_event_id": latest_event_id,
                "latest_event_status": latest_event_status,
                "latest_clip_path": latest_clip_path,
                "latest_stage3_label": latest_stage3_label,
                "latest_stage3_prob": _safe_float(latest_stage3_prob, default=0.0),
                "queue_status": queue_status,
                "queue_reason": queue_reason,
                "queue_size": _safe_int(queue_size, default=0) if queue_size != "" else "",
                "queue_capacity": _safe_int(queue_capacity, default=0) if queue_capacity != "" else "",
            }
        )

    recent_events = list(reversed(event_rows[-10:]))
    recent_stage3 = list(reversed(stage3_rows[-10:]))
    recent_status = list(reversed(status_rows[-20:]))

    return {
        "running": process_alive,
        "pid": pid,
        "return_code": return_code,
        "started_at": started_at,
        "run_dir": str(run_dir),
        "source_count": len(sources),
        "camera_count": len(cameras),
        "sources": sources,
        "cameras": cameras,
        "recent_events": recent_events,
        "recent_stage3": recent_stage3,
        "recent_status": recent_status,
    }