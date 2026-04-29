from django.shortcuts import redirect
from django.urls import reverse


class AuthRequiredMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response
        self.public_paths = [
            "/accounts/splash/",
            "/accounts/login/",
            "/accounts/admin-login/",
            "/accounts/register/",
            "/accounts/logout/",
            "/accounts/teacher-register/",
            "/accounts/sifremi-unuttum/",
            "/accounts/sifre-yenile/",
            "/static/",
            "/media/",
            "/djadmin/login/",
            "/favicon.ico",
        ]

    def __call__(self, request):
        path = request.path

        is_public = any(path.startswith(p) for p in self.public_paths)

        if not request.user.is_authenticated and not is_public:

            if (
                path.startswith("/admin/")
                or path.startswith("/panel/")
                or path.startswith("/admin-panel/")
            ):
                return redirect(
                    f"{reverse('accounts:admin_login')}?next={path}"
                )

            return redirect(
                f"{reverse('accounts:splash')}?next={path}"
            )

        response = self.get_response(request)

        if request.user.is_authenticated and not is_public:
            response["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
            response["Pragma"] = "no-cache"
            response["Expires"] = "0"

        return response