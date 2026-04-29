import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


BASE_DIR = Path(__file__).resolve().parent.parent

if load_dotenv:
    load_dotenv(BASE_DIR.parent.parent / ".env")

SECRET_KEY = "django-insecure-mezbq+2shqd7bs%@i(c)xp-yh=lu=e3_q$-vw3(+w+k9ie3=jn"

DEBUG = True

ALLOWED_HOSTS = []


INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",

    "streams",
    "accounts",
    "guvenlik",
    "adminx",
]


MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "accounts.middleware.AuthRequiredMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]


ROOT_URLCONF = "backend_frontend_project.urls"


TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]


WSGI_APPLICATION = "backend_frontend_project.wsgi.application"


DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
    }
}


AUTH_PASSWORD_VALIDATORS = [
    {
        "NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.MinimumLengthValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.CommonPasswordValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.NumericPasswordValidator",
    },
]


LANGUAGE_CODE = "tr-tr"
TIME_ZONE = "Europe/Istanbul"
USE_I18N = True
USE_TZ = True


STATIC_URL = "/static/"

STATICFILES_DIRS = [
    BASE_DIR / "static",
]


MEDIA_URL = "/media/"
MEDIA_ROOT = BASE_DIR / "media"


DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"


LOGIN_URL = "/accounts/login/"
LOGIN_REDIRECT_URL = "/"
LOGOUT_REDIRECT_URL = "/accounts/login/"


SESSION_COOKIE_AGE = 60 * 60 * 8
SESSION_EXPIRE_AT_BROWSER_CLOSE = False
SESSION_SAVE_EVERY_REQUEST = False
SESSION_COOKIE_HTTPONLY = True
SESSION_COOKIE_SAMESITE = "Lax"
CSRF_COOKIE_SAMESITE = "Lax"


REPO_ROOT = BASE_DIR.parent.parent

PIPELINE_OUTPUT_BASE = BASE_DIR / "media" / "pipeline_runs"

PIPELINE_ENTRY_MODULE = "fight.pipeline.run_multi_live_queue"

PIPELINE_DEFAULTS = {
    "motion_config": "fight/motion/configs/motion.yaml",
    "yolo_config": "fight/yolo/configs/yolo.yaml",
    "yolo_weights": "fight/yolo11n.pt",
    "pose_config": "fight/pose/configs/pose.yaml",
    "stage3_config": "fight/3D_CNN/configs/stage3.yaml",

    "person_conf": "0.25",
    "yolo_stride": "2",
    "pose_stride": "2",
    "fight_thr": "0.50",

    "use_pose": True,
    "use_stage3": True,

    "roi_size": "320",
    "min_persons_for_pose": "2",
    "pose_hold_frames": "8",
    "event_close_grace_frames": "12",
    "prebuffer_frames": "12",
    "max_event_frames": "160",
    "clip_fps": "16.0",
    "reconnect_sec": "1.0",
    "status_log_every": "30",

    "min_queue_frames": "32",
    "stage3_queue_size": "64",

    "preview_every_frames": "4",
    "preview_write_interval_sec": "0.50",
    "preview_jpeg_quality": "80",
    "clip_writer_queue_size": "32",
    "report_flush_interval_sec": "0.25",
    "cv2_threads": "1",

    "incident_enter_thr": "0.62",
    "incident_keep_thr": "0.60",
    "incident_vote_window": "7",
    "incident_vote_enter_needed": "2",
    "incident_vote_keep_needed": "2",
    "incident_merge_gap_sec": "10.0",
    "incident_max_bridge_nonfight": "2",
    "incident_min_segments": "2",
    "incident_single_strong_fight_thr": "0.82",
    "incident_confirm_min_duration_sec": "0.8",
    "incident_cooldown_sec": "18.0",
    "incident_clip_ready_wait_sec": "8.0",
    "incident_stale_finalize_sec": "4.0",
    "incident_temporal_iou_merge_thr": "0.42",
    "incident_write_nonfight": False,
}


# ------------------------------------------------------------
# External Mail Service Settings
# ------------------------------------------------------------

MAIL_SERVICE_ENABLED = os.getenv("MAIL_SERVICE_ENABLED", "True") == "True"

MAIL_API_URL = os.getenv(
    "MAIL_API_URL",
    "https://yzdd.gop.edu.tr/postman/api/v1/email/send",
)

MAIL_API_KEY = os.getenv("MAIL_API_KEY", "")

FRONTEND_URL = os.getenv(
    "FRONTEND_URL",
    "http://127.0.0.1:8000",
)
MAIL_LOGO_URL = os.getenv(
    "MAIL_LOGO_URL",
    f"{FRONTEND_URL}/static/images/togu-logo.png",
)