from django.db import models


class Camera(models.Model):
    name = models.CharField(max_length=120)
    camera_id = models.CharField(max_length=100, unique=True)
    source = models.CharField(max_length=500)
    description = models.TextField(blank=True)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.name} ({self.camera_id})"