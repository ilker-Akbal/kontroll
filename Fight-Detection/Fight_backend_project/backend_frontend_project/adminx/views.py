from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.db.models import Count, OuterRef, Subquery
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.cache import never_cache
from django.views.decorators.http import require_POST
from services.email_service import send_email, EmailServiceError
from accounts.decorators import role_required
from accounts.models import LoginActivity, UserProfile
from streams.models import Camera
from .forms import CameraForm, UserEditForm
from .forms import CameraForm


@never_cache
@login_required
@role_required(["admin"])
def dashboard(request):
    camera_count = Camera.objects.count()
    active_camera_count = Camera.objects.filter(is_active=True).count()
    passive_camera_count = Camera.objects.filter(is_active=False).count()

    total_user_count = User.objects.count()

    approved_user_count = UserProfile.objects.filter(status="approved").count()
    pending_user_count = UserProfile.objects.filter(status="pending").count()
    rejected_user_count = UserProfile.objects.filter(status="rejected").count()

    admin_user_count = UserProfile.objects.filter(role="admin").count()
    operator_user_count = UserProfile.objects.filter(role="operator").count()
    viewer_user_count = UserProfile.objects.filter(role="viewer").count()

    context = {
        "camera_count": camera_count,
        "active_camera_count": active_camera_count,
        "passive_camera_count": passive_camera_count,

        "user_count": UserProfile.objects.count(),
        "total_user_count": total_user_count,
        "approved_user_count": approved_user_count,
        "pending_user_count": pending_user_count,
        "rejected_user_count": rejected_user_count,

        "admin_user_count": admin_user_count,
        "operator_user_count": operator_user_count,
        "viewer_user_count": viewer_user_count,
    }

    return render(request, "adminx/dashboard.html", context)


@never_cache
@login_required
@role_required(["admin"])
def user_list(request):
    latest_login_activity = LoginActivity.objects.filter(
        user=OuterRef("user")
    ).order_by("-login_at")

    users = (
        UserProfile.objects.select_related("user")
        .annotate(
            login_count=Count("user__login_activities"),
            last_ip=Subquery(latest_login_activity.values("ip_address")[:1]),
            last_role_at_login=Subquery(
                latest_login_activity.values("role_at_login")[:1]
            ),
            last_activity_at=Subquery(
                latest_login_activity.values("login_at")[:1]
            ),
        )
        .order_by("-user__last_login", "-user__date_joined")
    )

    return render(
        request,
        "adminx/user_list.html",
        {
            "users": users,
            "user_count": UserProfile.objects.count(),
            "approved_user_count": UserProfile.objects.filter(status="approved").count(),
            "pending_user_count": UserProfile.objects.filter(status="pending").count(),
            "rejected_user_count": UserProfile.objects.filter(status="rejected").count(),
        },
    )


@never_cache
@login_required
@role_required(["admin"])
@require_POST
def user_approve(request, pk):
    profile = get_object_or_404(UserProfile, pk=pk)
    profile.status = "approved"
    profile.save()

    try:
        send_email(
            to=profile.user.email,
            subject="Hesabınız Onaylandı",
            body=f"""
            <h2>Hesabınız Onaylandı</h2>
            <p>Merhaba,</p>
            <p>Fight Detection sistemine erişim başvurunuz admin tarafından onaylandı.</p>
            <p>Artık sisteme giriş yapabilirsiniz.</p>
            <p><strong>Kullanıcı:</strong> {profile.user.username}</p>
            """
        )
    except EmailServiceError as exc:
        print(f"Onay maili gönderilemedi: {exc}")

    return redirect("adminx:user_list")

@never_cache
@login_required
@role_required(["admin"])
@require_POST
def user_reject(request, pk):
    profile = get_object_or_404(UserProfile, pk=pk)
    profile.status = "rejected"
    profile.save()

    try:
        send_email(
            to=profile.user.email,
            subject="Hesap Başvurunuz Reddedildi",
            body=f"""
            <h2>Hesap Başvurunuz Reddedildi</h2>
            <p>Merhaba,</p>
            <p>Fight Detection sistemine erişim başvurunuz admin tarafından reddedildi.</p>
            <p>Detaylı bilgi için sistem yöneticisiyle iletişime geçebilirsiniz.</p>
            <p><strong>Kullanıcı:</strong> {profile.user.username}</p>
            """
        )
    except EmailServiceError as exc:
        print(f"Red maili gönderilemedi: {exc}")

    return redirect("adminx:user_list")

@never_cache
@login_required
@role_required(["admin"])
def user_delete(request, pk):
    profile = get_object_or_404(
        UserProfile.objects.select_related("user"),
        pk=pk
    )

    if request.method == "POST":
        profile.user.delete()
        return redirect("adminx:user_list")

    return render(
        request,
        "adminx/user_delete.html",
        {"profile": profile},
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

@never_cache
@login_required
@role_required(["admin"])
def user_update(request, pk):
    profile = get_object_or_404(
        UserProfile.objects.select_related("user"),
        pk=pk
    )

    form = UserEditForm(
        request.POST or None,
        instance=profile,
        user_instance=profile.user
    )

    if request.method == "POST" and form.is_valid():
        form.save(user_instance=profile.user)
        return redirect("adminx:user_list")

    return render(
        request,
        "adminx/user_form.html",
        {
            "form": form,
            "page_title": "Kullanıcı Düzenle",
            "submit_label": "Güncelle",
            "profile": profile,
        },
    )