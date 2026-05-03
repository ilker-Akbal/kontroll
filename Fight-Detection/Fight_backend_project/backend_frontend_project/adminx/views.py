from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.db.models import Count, OuterRef, Subquery
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.cache import never_cache
from django.views.decorators.http import require_POST
from pathlib import Path
from django.conf import settings
from services.pipeline_bridge.report_reader import build_dashboard_report
from accounts.decorators import role_required
from accounts.models import LoginActivity, UserProfile
from services.email_service import EmailServiceError, send_email
from streams.models import Camera
from django.core.paginator import Paginator
from pathlib import Path
from django.conf import settings
from services.pipeline_bridge.report_reader import build_dashboard_report
from .forms import CameraForm, UserEditForm
from django.conf import settings

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

    login_url = f"{settings.FRONTEND_URL}/accounts/login/"
    logo_url = getattr(
        settings,
        "MAIL_LOGO_URL",
        f"{settings.FRONTEND_URL}/static/images/togu-logo.png",
    )

    mail_body = f"""
    <div style="margin:0;padding:0;background:#f4f7fb;font-family:Arial,sans-serif;">
      <div style="max-width:640px;margin:0 auto;padding:34px 18px;">
        <div style="background:#ffffff;border-radius:24px;padding:36px 30px;border:1px solid #e5e7eb;box-shadow:0 10px 30px rgba(15,23,42,0.08);text-align:center;">

          <div style="margin-bottom:24px;">
            <img
              src="{logo_url}"
              alt="TOGÜ Logo"
              style="width:120px;height:auto;margin:0 auto;display:block;"
            />
          </div>

          <div style="width:58px;height:58px;border-radius:18px;background:#dcfce7;color:#15803d;margin:0 auto 18px;display:flex;align-items:center;justify-content:center;font-size:28px;font-weight:900;">
            ✓
          </div>

          <h1 style="margin:0 0 12px;color:#0f172a;font-size:26px;font-weight:800;">
            Hesabınız Onaylandı
          </h1>

          <p style="margin:0 auto 22px;max-width:500px;color:#64748b;font-size:15px;line-height:1.75;">
            Fight Detection sistemine erişim başvurunuz admin tarafından onaylanmıştır.
            Artık kullanıcı bilgilerinizle sisteme giriş yapabilirsiniz.
          </p>

          <div style="margin:22px 0;padding:18px;border-radius:18px;background:#f8fafc;border:1px solid #e2e8f0;text-align:left;">
            <div style="margin-bottom:10px;">
              <span style="display:block;color:#64748b;font-size:12px;font-weight:700;">Kullanıcı Adı</span>
              <strong style="display:block;color:#0f172a;font-size:15px;font-weight:800;">
                {profile.user.username}
              </strong>
            </div>

            <div style="margin-bottom:10px;">
              <span style="display:block;color:#64748b;font-size:12px;font-weight:700;">E-posta</span>
              <strong style="display:block;color:#0f172a;font-size:15px;font-weight:800;">
                {profile.user.email or "-"}
              </strong>
            </div>

            <div>
              <span style="display:block;color:#64748b;font-size:12px;font-weight:700;">Rol</span>
              <strong style="display:block;color:#0f172a;font-size:15px;font-weight:800;">
                {profile.get_role_display()}
              </strong>
            </div>
          </div>

          <a
            href="{login_url}"
            style="display:inline-block;background:#0f4c81;color:#ffffff;text-decoration:none;padding:14px 28px;border-radius:14px;font-size:15px;font-weight:800;box-shadow:0 12px 24px rgba(15,76,129,0.22);"
          >
            Sisteme Giriş Yap
          </a>

          <p style="margin:24px 0 0;color:#94a3b8;font-size:12px;line-height:1.6;">
            Bu e-posta Fight Detection sistemi tarafından otomatik olarak gönderilmiştir.
            Eğer bu işlem hakkında bilginiz yoksa sistem yöneticisiyle iletişime geçiniz.
          </p>

        </div>
      </div>
    </div>
    """

    try:
        send_email(
            to=profile.user.email,
            subject="Fight Detection Hesabınız Onaylandı",
            body=mail_body,
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
        pk=pk,
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
def user_update(request, pk):
    profile = get_object_or_404(
        UserProfile.objects.select_related("user"),
        pk=pk,
    )

    form = UserEditForm(
        request.POST or None,
        instance=profile,
        user_instance=profile.user,
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

MAX_ADMIN_INCIDENT_RUNS = 30
MAX_ADMIN_INCIDENT_ROWS = 500
ADMIN_INCIDENTS_PER_PAGE = 6


def _admin_pipeline_runs_root() -> Path:
    return Path(settings.MEDIA_ROOT) / "pipeline_runs"


def _admin_run_dirs(limit: int = MAX_ADMIN_INCIDENT_RUNS) -> list[Path]:
    root = _admin_pipeline_runs_root()

    if not root.exists() or not root.is_dir():
        return []

    dirs = [p for p in root.iterdir() if p.is_dir()]
    dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    return dirs[:limit]


def _admin_active_sources():
    return [
        {
            "camera_id": cam.camera_id,
            "source": cam.source,
            "name": cam.name,
            "description": cam.description,
        }
        for cam in Camera.objects.filter(is_active=True).order_by("-created_at")
    ]


def _admin_format_ts(value):
    if value in (None, "", "-"):
        return "-"

    s = str(value).strip().replace("T", " ")

    if "." in s:
        s = s.split(".", 1)[0]

    return s


def _admin_read_run_report(run_dir: Path):
    try:
        report = build_dashboard_report(
            run_dir=str(run_dir),
            sources=_admin_active_sources(),
            process_alive=False,
            pid=None,
            started_at=None,
            return_code=None,
            media_root=settings.MEDIA_ROOT,
        )

        report["run_name"] = report.get("run_name") or run_dir.name
        return report

    except Exception:
        return None


def _admin_incident_clip_url(run_name, clip_path):
    if not run_name or not clip_path:
        return ""

    clip_name = Path(str(clip_path)).name

    return f"/dashboard/incident-video/{run_name}/{clip_name}/"


def _admin_faculty_users_map():
    profiles = (
        UserProfile.objects
        .select_related("user")
        .filter(status="approved")
        .order_by("faculty", "user__username")
    )

    out = {}

    for profile in profiles:
        if not profile.faculty:
            continue

        out.setdefault(profile.faculty, []).append(
            {
                "username": profile.user.username,
                "email": profile.user.email,
                "role": profile.get_role_display(),
            }
        )

    return out


def _admin_collect_incidents():
    camera_map = {
        cam.camera_id: cam
        for cam in Camera.objects.all()
    }

    faculty_users = _admin_faculty_users_map()

    rows = []
    seen = set()

    for run_dir in _admin_run_dirs():
        report = _admin_read_run_report(run_dir)

        if not report:
            continue

        run_name = report.get("run_name") or run_dir.name

        for row in report.get("recent_incidents", []):
            camera_id = row.get("camera_id") or "-"
            incident_id = row.get("incident_id") or "-"
            clip_path = row.get("clip_path") or ""

            key = (
                run_name,
                camera_id,
                incident_id,
                clip_path,
            )

            if key in seen:
                continue

            seen.add(key)

            camera = camera_map.get(camera_id)
            faculty_value = camera.faculty if camera else None
            faculty_label = camera.get_faculty_display() if camera and camera.faculty else "-"

            rows.append(
                {
                    "run_name": run_name,
                    "camera_id": camera_id,
                    "camera_name": camera.name if camera else "-",
                    "camera_source": camera.source if camera else "-",
                    "faculty": faculty_value or "",
                    "faculty_label": faculty_label,
                    "faculty_users": faculty_users.get(faculty_value, []),
                    "incident_id": incident_id,
                    "start_ts": _admin_format_ts(row.get("start_ts")),
                    "end_ts": _admin_format_ts(row.get("end_ts")),
                    "final_label": row.get("final_label") or "-",
                    "clip_path": clip_path,
                    "clip_url": _admin_incident_clip_url(run_name, clip_path),
                    "part_count": row.get("part_count", "-"),
                }
            )

    rows.sort(
        key=lambda x: (
            str(x.get("end_ts") or ""),
            str(x.get("start_ts") or ""),
            str(x.get("incident_id") or ""),
        ),
        reverse=True,
    )

    return rows[:MAX_ADMIN_INCIDENT_ROWS]


def _filter_admin_incidents(incidents, selected_faculty, selected_camera, search_query):
    filtered = incidents

    if selected_faculty:
        filtered = [
            item for item in filtered
            if item.get("faculty") == selected_faculty
        ]

    if selected_camera:
        filtered = [
            item for item in filtered
            if item.get("camera_id") == selected_camera
        ]

    if search_query:
        q = search_query.lower()

        filtered = [
            item for item in filtered
            if q in str(item.get("incident_id", "")).lower()
            or q in str(item.get("run_name", "")).lower()
            or q in str(item.get("camera_id", "")).lower()
            or q in str(item.get("camera_name", "")).lower()
            or q in str(item.get("faculty_label", "")).lower()
            or q in str(item.get("final_label", "")).lower()
        ]

    return filtered


@never_cache
@login_required
@role_required(["admin"])
def incident_list(request):
    selected_faculty = request.GET.get("faculty", "").strip()
    selected_camera = request.GET.get("camera", "").strip()
    search_query = request.GET.get("q", "").strip()

    all_incidents = _admin_collect_incidents()

    filtered_incidents = _filter_admin_incidents(
        incidents=all_incidents,
        selected_faculty=selected_faculty,
        selected_camera=selected_camera,
        search_query=search_query,
    )

    fight_count = sum(
        1 for item in filtered_incidents
        if str(item.get("final_label", "")).lower() == "fight"
    )

    faculty_count = len(
        {
            item.get("faculty")
            for item in filtered_incidents
            if item.get("faculty")
        }
    )

    paginator = Paginator(filtered_incidents, ADMIN_INCIDENTS_PER_PAGE)
    page_number = request.GET.get("page")
    page_obj = paginator.get_page(page_number)

    query_params = request.GET.copy()

    if "page" in query_params:
        query_params.pop("page")

    return render(
        request,
        "adminx/incident_list.html",
        {
            "incidents": page_obj.object_list,
            "page_obj": page_obj,
            "paginator": paginator,

            "incident_count": len(filtered_incidents),
            "all_incident_count": len(all_incidents),
            "fight_count": fight_count,
            "faculty_count": faculty_count,

            "selected_faculty": selected_faculty,
            "selected_camera": selected_camera,
            "search_query": search_query,
            "query_string": query_params.urlencode(),

            "faculty_options": UserProfile.FACULTY_CHOICES,
            "camera_options": Camera.objects.all().order_by("name", "camera_id"),
        },
    )