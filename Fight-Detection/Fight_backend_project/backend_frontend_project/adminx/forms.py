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
from django import forms
from django.contrib.auth.models import User
from accounts.models import UserProfile


class UserEditForm(forms.ModelForm):
    username = forms.CharField(label="Kullanıcı Adı")
    email = forms.EmailField(label="E-posta", required=False)

    class Meta:
        model = UserProfile
        fields = ["faculty", "role", "status"]

    def __init__(self, *args, **kwargs):
        user = kwargs.pop("user_instance")
        super().__init__(*args, **kwargs)

        self.fields["username"].initial = user.username
        self.fields["email"].initial = user.email

    def save(self, user_instance, commit=True):
        user_instance.username = self.cleaned_data["username"]
        user_instance.email = self.cleaned_data["email"]

        if commit:
            user_instance.save()

        return super().save(commit=commit)