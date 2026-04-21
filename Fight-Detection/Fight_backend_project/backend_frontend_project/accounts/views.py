from django.contrib.auth import authenticate, login, logout
from django.shortcuts import render, redirect
from django.views.decorators.http import require_http_methods, require_POST
from django.views.decorators.cache import never_cache

from .models import UserProfile, LoginActivity
from .forms import UserRegisterForm


def _safe_next_url(next_url):
    next_url = (next_url or "").strip()

    if not next_url.startswith("/"):
        return ""

    if next_url.startswith("//"):
        return ""

    if next_url.startswith("/admin/"):
        return next_url

    if next_url.startswith("/dashboard/"):
        return next_url

    return ""


def _get_client_ip(request):
    ip = request.META.get("HTTP_X_FORWARDED_FOR")
    if ip:
        return ip.split(",")[0].strip()
    return request.META.get("REMOTE_ADDR")


def _create_login_activity(request, user, profile):
    LoginActivity.objects.create(
        user=user,
        role_at_login=profile.role,
        ip_address=_get_client_ip(request),
        user_agent=request.META.get("HTTP_USER_AGENT", ""),
    )


@never_cache
@require_http_methods(["GET", "POST"])
def login_view(request):
    next_url = _safe_next_url(
        request.POST.get("next") or request.GET.get("next") or "/dashboard/"
    )

    if request.user.is_authenticated:
        return redirect("/dashboard/")

    error = None

    if request.method == "POST":
        username = request.POST.get("username", "").strip()
        password = request.POST.get("password", "").strip()

        if not username or not password:
            error = "Kullanıcı adı ve şifre zorunludur."

        else:
            user = authenticate(request, username=username, password=password)

            if user is None:
                error = "Kullanıcı adı veya şifre hatalı."

            elif not user.is_active:
                error = "Bu hesap pasif durumda."

            else:
                profile, _ = UserProfile.objects.get_or_create(user=user)

                if profile.status == "pending":
                    error = "Hesabınız admin onayı bekliyor."

                elif profile.status == "rejected":
                    error = "Hesabınız reddedildi."

                else:
                    login(request, user)
                    request.session.set_expiry(60 * 60 * 8)
                    _create_login_activity(request, user, profile)

                    if next_url:
                        return redirect(next_url)

                    return redirect("/dashboard/")

    return render(
        request,
        "accounts/login.html",
        {
            "error": error,
            "next_url": next_url,
        },
    )


@never_cache
@require_http_methods(["GET", "POST"])
def admin_login_view(request):
    next_url = _safe_next_url(
        request.POST.get("next") or request.GET.get("next") or "/admin/"
    )

    if request.user.is_authenticated:
        return redirect("/admin/")

    error = None

    if request.method == "POST":
        username = request.POST.get("username", "").strip()
        password = request.POST.get("password", "").strip()

        if not username or not password:
            error = "Kullanıcı adı ve şifre zorunludur."

        else:
            user = authenticate(request, username=username, password=password)

            if user is None:
                error = "Kullanıcı adı veya şifre hatalı."

            elif not user.is_active:
                error = "Bu hesap pasif durumda."

            else:
                profile, _ = UserProfile.objects.get_or_create(user=user)

                if user.is_superuser or user.is_staff:
                    profile.role = "admin"
                    profile.status = "approved"
                    profile.save()

                if profile.status == "pending":
                    error = "Hesabınız admin onayı bekliyor."

                elif profile.status == "rejected":
                    error = "Hesabınız reddedildi."

                elif profile.role != "admin":
                    error = "Bu alan yalnızca admin içindir."

                else:
                    login(request, user)
                    request.session.set_expiry(60 * 60 * 8)
                    _create_login_activity(request, user, profile)

                    if next_url:
                        return redirect(next_url)

                    return redirect("/admin/")

    return render(
        request,
        "accounts/admin_login.html",
        {
            "error": error,
            "next_url": next_url,
        },
    )


@never_cache
@require_http_methods(["GET", "POST"])
def register_view(request):
    form = UserRegisterForm(request.POST or None)

    if request.method == "POST":
        if form.is_valid():
            form.save()
            return redirect("accounts:login")

    return render(
        request,
        "accounts/register.html",
        {
            "form": form,
        },
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