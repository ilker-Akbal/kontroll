from django.urls import path
from .views import (
    login_view,
    admin_login_view,
    register_view,
    logout_view,
)

app_name = "accounts"

urlpatterns = [
    path("login/", login_view, name="login"),
    path("register/", register_view, name="register"),
    path("admin-login/", admin_login_view, name="admin_login"),
    path("logout/", logout_view, name="logout"),
]