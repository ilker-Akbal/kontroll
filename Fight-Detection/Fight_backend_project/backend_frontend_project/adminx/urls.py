from django.urls import path
from .views import dashboard, camera_list, camera_create, camera_edit, camera_delete

app_name = "adminx"

urlpatterns = [
    path("", dashboard, name="dashboard"),
    path("cameras/", camera_list, name="camera_list"),
    path("cameras/create/", camera_create, name="camera_create"),
    path("cameras/<int:pk>/edit/", camera_edit, name="camera_edit"),
    path("cameras/<int:pk>/delete/", camera_delete, name="camera_delete"),
]