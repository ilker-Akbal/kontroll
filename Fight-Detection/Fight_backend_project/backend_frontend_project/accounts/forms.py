from django import forms
from django.contrib.auth.models import User
from .models import UserProfile


class UserRegisterForm(forms.Form):
    faculty = forms.ChoiceField(
        choices=UserProfile.FACULTY_CHOICES,
        widget=forms.Select()
    )
    email = forms.EmailField(
        widget=forms.EmailInput(
            attrs={"placeholder": "ornek@togu.edu.tr"}
        )
    )
    password1 = forms.CharField(
        widget=forms.PasswordInput(
            attrs={"placeholder": "Şifrenizi girin"}
        )
    )
    password2 = forms.CharField(
        widget=forms.PasswordInput(
            attrs={"placeholder": "Şifrenizi tekrar girin"}
        )
    )

    def clean_email(self):
        email = self.cleaned_data.get("email")
        if User.objects.filter(email=email).exists():
            raise forms.ValidationError("Bu mail adresi zaten kayıtlı.")
        return email

    def clean(self):
        cleaned_data = super().clean()
        password1 = cleaned_data.get("password1")
        password2 = cleaned_data.get("password2")

        if password1 and password2 and password1 != password2:
            self.add_error("password2", "Şifreler eşleşmiyor.")

        return cleaned_data

    def save(self):
        email = self.cleaned_data["email"]
        password = self.cleaned_data["password1"]
        faculty = self.cleaned_data["faculty"]

        user = User.objects.create_user(
            username=email,
            email=email,
            password=password
        )

        user.profile.faculty = faculty
        user.profile.save()

        return user