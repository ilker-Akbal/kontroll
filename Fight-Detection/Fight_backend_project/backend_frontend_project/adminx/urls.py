from django.urls import path
from .views import (
    dashboard,
    user_list,
    user_approve,
    user_reject,
    user_delete,
    camera_list,
    camera_create,
    camera_edit,
    camera_delete,
    user_update,
    incident_list
)

app_name = "adminx"

urlpatterns = [
    path("", dashboard, name="dashboard"),

    path("users/", user_list, name="user_list"),
    path("users/<int:pk>/approve/", user_approve, name="user_approve"),
    path("users/<int:pk>/reject/", user_reject, name="user_reject"),
    path("users/<int:pk>/delete/", user_delete, name="user_delete"),

    path("cameras/", camera_list, name="camera_list"),
    path("cameras/create/", camera_create, name="camera_create"),
    path("cameras/<int:pk>/edit/", camera_edit, name="camera_edit"),
    path("cameras/<int:pk>/delete/", camera_delete, name="camera_delete"),
    path("users/<int:pk>/edit/", user_update, name="user_update"),
    path("incidents/", incident_list, name="incident_list"),
]