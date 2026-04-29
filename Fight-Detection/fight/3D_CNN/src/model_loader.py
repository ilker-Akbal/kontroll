from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def build_x3dm(num_classes: int = 2, pretrained: bool = False) -> torch.nn.Module:
    """
    X3D-M modelini production için kurar.

    Not:
    - Canlı sistemde torch.hub kullanmıyoruz.
    - pytorchvideo kurulu olmalı.
    - Eğitim kodundaki mimari:
        model = x3d_m(pretrained=True)
        model.blocks[-1].proj = nn.Linear(..., 2)
    """
    try:
        from pytorchvideo.models.hub import x3d_m
    except Exception as exc:
        raise RuntimeError(
            "X3D-M için pytorchvideo kurulu olmalı. "
            "Kurulum: pip install pytorchvideo fvcore iopath"
        ) from exc

    model = x3d_m(pretrained=pretrained)

    if not hasattr(model, "blocks") or not hasattr(model.blocks[-1], "proj"):
        raise RuntimeError("Beklenen X3D-M head yapısı bulunamadı: model.blocks[-1].proj")

    in_features = model.blocks[-1].proj.in_features
    model.blocks[-1].proj = nn.Linear(in_features, int(num_classes))
    return model


def build_model(
    num_classes: int = 2,
    model_name: str = "x3d_m",
    pretrained: bool = False,
) -> torch.nn.Module:
    name = str(model_name or "x3d_m").lower().strip()

    if name in {"x3d_m", "x3dm", "x3d-m"}:
        return build_x3dm(num_classes=num_classes, pretrained=pretrained)

    raise ValueError(
        f"Desteklenmeyen stage3 model_name={model_name!r}. "
        "Bu sistem artık production için x3d_m kullanacak şekilde ayarlandı."
    )


def _resolve_ckpt_path(ckpt_path: str) -> str:
    p = Path(str(ckpt_path))

    if p.is_absolute() and p.exists():
        return str(p)

    cwd_cand = (Path.cwd() / p).resolve()
    if cwd_cand.exists():
        return str(cwd_cand)

    stage3_root = Path(__file__).resolve().parents[1]

    cand1 = (stage3_root / p).resolve()
    if cand1.exists():
        return str(cand1)

    cand2 = (stage3_root / "weights" / p.name).resolve()
    if cand2.exists():
        return str(cand2)

    raise FileNotFoundError(f"Checkpoint bulunamadı: {ckpt_path}")


def _is_lfs_pointer(path: str) -> bool:
    head = Path(path).read_bytes()[:128]
    return head.startswith(b"version https://git-lfs.github.com/spec/v1")


def _is_zip(path: str) -> bool:
    try:
        return zipfile.is_zipfile(path)
    except Exception:
        return False


def _extract_state_dict(ckpt: Any) -> dict:
    """
    Eğitim kodundaki kayıt formatını destekler:
        {
            "model": state_dict,
            "optimizer": ...,
            "scheduler": ...,
            "model_name": "x3d_m",
            ...
        }

    Ayrıca klasik state_dict / model_state_dict formatlarını da destekler.
    """
    if isinstance(ckpt, dict):
        for key in ("model", "state_dict", "model_state_dict"):
            val = ckpt.get(key)
            if isinstance(val, dict):
                return val

    if isinstance(ckpt, dict):
        # Direkt state_dict olabilir.
        tensor_like = [k for k, v in ckpt.items() if hasattr(v, "shape")]
        if tensor_like:
            return ckpt

    raise RuntimeError(
        "Checkpoint içinden model ağırlıkları okunamadı. "
        "Beklenen key: 'model', 'state_dict' veya 'model_state_dict'."
    )


def _clean_state_dict(sd: dict) -> dict:
    """
    torch.compile / DataParallel kaynaklı prefixleri temizler.
    """
    cleaned = {}

    for k, v in sd.items():
        nk = str(k)

        prefixes = (
            "module.",
            "_orig_mod.",
            "model.",
        )

        changed = True
        while changed:
            changed = False
            for pref in prefixes:
                if nk.startswith(pref):
                    nk = nk[len(pref):]
                    changed = True

        cleaned[nk] = v

    return cleaned


def load_ckpt(
    model: torch.nn.Module,
    ckpt_path: str,
    strict: bool = True,
):
    ckpt_path = _resolve_ckpt_path(ckpt_path)

    if _is_lfs_pointer(ckpt_path):
        raise RuntimeError(
            f"Checkpoint gerçek ağırlık değil, Git LFS pointer dosyası: {ckpt_path}"
        )

    if _is_zip(ckpt_path):
        # torch.save yeni zip formatı zaten zip görünebilir, bu hata değildir.
        pass

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = _clean_state_dict(_extract_state_dict(ckpt))

    missing, unexpected = model.load_state_dict(sd, strict=False)

    if strict and (missing or unexpected):
        msg = [
            f"Checkpoint model ile tam uyuşmadı: {ckpt_path}",
            f"missing_keys={len(missing)}",
            f"unexpected_keys={len(unexpected)}",
        ]

        if missing:
            msg.append("İlk missing keys: " + ", ".join(missing[:20]))
        if unexpected:
            msg.append("İlk unexpected keys: " + ", ".join(unexpected[:20]))

        raise RuntimeError("\n".join(msg))

    return ckpt


def load_model(
    ckpt_path: str,
    device: str = "cuda",
    num_classes: int = 2,
    model_name: str = "x3d_m",
    pretrained: bool = False,
    strict: bool = True,
) -> torch.nn.Module:
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    model = build_model(
        num_classes=int(num_classes),
        model_name=model_name,
        pretrained=bool(pretrained),
    )

    load_ckpt(model, ckpt_path, strict=bool(strict))

    model.eval()
    model.to(device)
    return model