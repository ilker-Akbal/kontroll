from __future__ import annotations

import cv2


def save_clip_mp4(frames_bgr, out_path: str, fps: float = 16.0):
    if not frames_bgr:
        return

    h, w = frames_bgr[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    try:
        for f in frames_bgr:
            writer.write(f)
    finally:
        writer.release()