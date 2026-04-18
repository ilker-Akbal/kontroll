from django import forms
from streams.models import Camera


class CameraForm(forms.ModelForm):
    class Meta:
        model = Camera
        fields = [
            "name",
            "camera_id",
            "source",
            "description",
            "is_active",
        ]