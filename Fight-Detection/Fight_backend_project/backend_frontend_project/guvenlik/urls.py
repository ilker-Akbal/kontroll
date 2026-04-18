from django.urls import path
from .views import index, status, stream

app_name = "dashboard"

urlpatterns = [
    path("", index, name="index"),
    path("status/", status, name="status"),
    path("stream/<str:camera_id>/", stream, name="stream"),
]