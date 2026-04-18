from django.urls import path
from .views import (
    login_view,
    admin_login_view,
    forgot_password,
    logout_view,
)

app_name = "accounts"

urlpatterns = [
    path("login/", login_view, name="login"),
    path("admin-login/", admin_login_view, name="admin_login"),
    path("logout/", logout_view, name="logout"),
    path("forgot-password/", forgot_password, name="forgot_password"),
]