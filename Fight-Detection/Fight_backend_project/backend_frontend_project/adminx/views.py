from django.contrib.auth.decorators import login_required
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.cache import never_cache

from accounts.decorators import role_required
from streams.models import Camera
from .forms import CameraForm


@never_cache
@login_required
@role_required(["admin"])
def dashboard(request):
    return render(
        request,
        "adminx/dashboard.html",
        {
            "camera_count": Camera.objects.count(),
            "active_camera_count": Camera.objects.filter(is_active=True).count(),
            "passive_camera_count": Camera.objects.filter(is_active=False).count(),
        },
    )


@never_cache
@login_required
@role_required(["admin"])
def camera_list(request):
    cameras = Camera.objects.all()
    return render(
        request,
        "adminx/camera_list.html",
        {"cameras": cameras},
    )


@never_cache
@login_required
@role_required(["admin"])
def camera_create(request):
    form = CameraForm(request.POST or None)

    if request.method == "POST" and form.is_valid():
        form.save()
        return redirect("adminx:camera_list")

    return render(
        request,
        "adminx/camera_form.html",
        {
            "form": form,
            "page_title": "Kamera Ekle",
            "submit_label": "Kaydet",
        },
    )


@never_cache
@login_required
@role_required(["admin"])
def camera_edit(request, pk):
    camera = get_object_or_404(Camera, pk=pk)
    form = CameraForm(request.POST or None, instance=camera)

    if request.method == "POST" and form.is_valid():
        form.save()
        return redirect("adminx:camera_list")

    return render(
        request,
        "adminx/camera_form.html",
        {
            "form": form,
            "page_title": "Kamera Düzenle",
            "submit_label": "Güncelle",
            "camera": camera,
        },
    )


@never_cache
@login_required
@role_required(["admin"])
def camera_delete(request, pk):
    camera = get_object_or_404(Camera, pk=pk)

    if request.method == "POST":
        camera.delete()
        return redirect("adminx:camera_list")

    return render(
        request,
        "adminx/camera_delete.html",
        {"camera": camera},
    )