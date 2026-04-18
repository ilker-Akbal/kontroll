from django.contrib.auth import authenticate, login, logout
from django.http import HttpResponse
from django.shortcuts import render, redirect
from django.views.decorators.http import require_http_methods, require_POST
from django.views.decorators.cache import never_cache

from .models import UserProfile


def _safe_next_url(next_url: str) -> str:
    next_url = (next_url or "").strip()

    if not next_url.startswith("/"):
        return ""

    if next_url.startswith("//"):
        return ""

    allowed_prefixes = [
        "/admin/",
        "/dashboard/",
    ]

    for prefix in allowed_prefixes:
        if next_url.startswith(prefix):
            return next_url

    if next_url == "/":
        return "/dashboard/"

    return ""


@never_cache
@require_http_methods(["GET", "POST"])
def login_view(request):
    next_url = _safe_next_url(
        request.POST.get("next") or request.GET.get("next") or "/dashboard/"
    )

    if request.user.is_authenticated:
        profile = getattr(request.user, "profile", None)
        if request.user.is_superuser or request.user.is_staff or (profile and profile.role == "admin"):
            return redirect("/admin/")
        return redirect("/dashboard/")

    error = None

    if request.method == "POST":
        username = request.POST.get("username", "").strip()
        password = request.POST.get("password", "").strip()

        if not username or not password:
            error = "Kullanıcı adı ve şifre zorunludur."
            return render(
                request,
                "accounts/login.html",
                {"error": error, "next_url": next_url},
            )

        user = authenticate(request, username=username, password=password)

        if user is None:
            error = "Kullanıcı adı veya şifre hatalı."
            return render(
                request,
                "accounts/login.html",
                {"error": error, "next_url": next_url},
            )

        if not user.is_active:
            error = "Bu hesap pasif durumda."
            return render(
                request,
                "accounts/login.html",
                {"error": error, "next_url": next_url},
            )

        profile = getattr(user, "profile", None)
        if profile is None:
            profile, _ = UserProfile.objects.get_or_create(user=user)

        if not profile.is_approved:
            error = "Hesabınız henüz onaylanmamış."
            return render(
                request,
                "accounts/login.html",
                {"error": error, "next_url": next_url},
            )

        login(request, user)
        request.session.set_expiry(60 * 60 * 8)

        if next_url.startswith("/dashboard/"):
            return redirect(next_url)

        return redirect("/dashboard/")

    return render(
        request,
        "accounts/login.html",
        {"error": error, "next_url": next_url},
    )


@never_cache
@require_http_methods(["GET", "POST"])
def admin_login_view(request):
    next_url = _safe_next_url(
        request.POST.get("next") or request.GET.get("next") or "/admin/"
    )

    if request.user.is_authenticated:
        profile = getattr(request.user, "profile", None)
        if request.user.is_superuser or request.user.is_staff or (profile and profile.role == "admin"):
            return redirect("/admin/")
        return redirect("/dashboard/")

    error = None

    if request.method == "POST":
        username = request.POST.get("username", "").strip()
        password = request.POST.get("password", "").strip()

        if not username or not password:
            error = "Kullanıcı adı ve şifre zorunludur."
            return render(
                request,
                "accounts/admin_login.html",
                {"error": error, "next_url": next_url},
            )

        user = authenticate(request, username=username, password=password)

        if user is None:
            error = "Kullanıcı adı veya şifre hatalı."
            return render(
                request,
                "accounts/admin_login.html",
                {"error": error, "next_url": next_url},
            )

        if not user.is_active:
            error = "Bu hesap pasif durumda."
            return render(
                request,
                "accounts/admin_login.html",
                {"error": error, "next_url": next_url},
            )

        profile = getattr(user, "profile", None)
        if profile is None:
            profile, _ = UserProfile.objects.get_or_create(user=user)

        # İlk admin bootstrap mantığı:
        # superuser/staff kullanıcı admin girişinden geçebilsin
        # ve profil rolü otomatik admin'e çekilsin
        if user.is_superuser or user.is_staff:
            if profile.role != "admin" or not profile.is_approved:
                profile.role = "admin"
                profile.is_approved = True
                profile.save()

        if not profile.is_approved:
            error = "Hesabınız henüz onaylanmamış."
            return render(
                request,
                "accounts/admin_login.html",
                {"error": error, "next_url": next_url},
            )

        if not (user.is_superuser or user.is_staff or profile.role == "admin"):
            error = "Bu ekran yalnızca admin kullanıcıları içindir."
            return render(
                request,
                "accounts/admin_login.html",
                {"error": error, "next_url": next_url},
            )

        login(request, user)
        request.session.set_expiry(60 * 60 * 8)

        if next_url.startswith("/admin/"):
            return redirect(next_url)

        return redirect("/admin/")

    return render(
        request,
        "accounts/admin_login.html",
        {"error": error, "next_url": next_url},
    )


@never_cache
@require_POST
def logout_view(request):
    logout(request)
    request.session.flush()
    response = redirect("accounts:login")
    response["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response["Pragma"] = "no-cache"
    response["Expires"] = "0"
    return response


def forgot_password(request):
    return HttpResponse("Bu sayfa Faz 1'de aktif değil.")