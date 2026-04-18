import time

import cv2
from django.contrib.auth.decorators import login_required
from django.http import Http404, JsonResponse, StreamingHttpResponse
from django.shortcuts import get_object_or_404, render
from django.views.decorators.cache import never_cache
from django.views.decorators.http import require_GET

from streams.models import Camera


@never_cache
@login_required
def index(request):
    cameras = Camera.objects.filter(is_active=True).order_by("-created_at")
    return render(request, "dashboard/index.html", {"cameras": cameras})


@never_cache
@login_required
@require_GET
def status(request):
    cameras = Camera.objects.filter(is_active=True).order_by("-created_at")

    data = {
        "count": cameras.count(),
        "cameras": [
            {
                "id": camera.id,
                "name": camera.name,
                "camera_id": camera.camera_id,
                "source": camera.source,
                "description": camera.description,
                "is_active": camera.is_active,
            }
            for camera in cameras
        ],
    }
    return JsonResponse(data)


def _open_capture(source: str):
    if source is None:
        return None

    source = str(source).strip()

    if source.isdigit():
        return cv2.VideoCapture(int(source))

    return cv2.VideoCapture(source)


def _mjpeg_generator(source: str):
    cap = _open_capture(source)
    if cap is None or not cap.isOpened():
        return

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.05)
                continue

            ok, buffer = cv2.imencode(".jpg", frame)
            if not ok:
                time.sleep(0.05)
                continue

            frame_bytes = buffer.tobytes()

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + frame_bytes
                + b"\r\n"
            )

            time.sleep(0.03)

    finally:
        cap.release()


@never_cache
@login_required
@require_GET
def stream(request, camera_id):
    camera = get_object_or_404(
        Camera,
        camera_id=camera_id,
        is_active=True,
    )

    cap = _open_capture(camera.source)
    if cap is None or not cap.isOpened():
        if cap is not None:
            cap.release()
        raise Http404("Kamera akışı açılamadı")

    cap.release()

    return StreamingHttpResponse(
        _mjpeg_generator(camera.source),
        content_type="multipart/x-mixed-replace; boundary=frame",
    )