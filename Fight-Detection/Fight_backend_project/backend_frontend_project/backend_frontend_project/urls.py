from django.contrib import admin
from django.urls import include, path
from django.shortcuts import redirect
from guvenlik.views import index

urlpatterns = [
    path("djadmin/", admin.site.urls),
    path("accounts/", include("accounts.urls")),
    path("admin/", include("adminx.urls")),
    path("panel/", lambda request: redirect("/admin/")),
    path("dashboard/", include("guvenlik.urls")),
    path("", index, name="home"),
]