from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from fight.pipeline.utils import resize_with_padding


class MotionAdapter:
    def __init__(self, cfg_path: str):
        from fight.motion.src.core.config import load_config
        from fight.motion.src.service.motion_service import MotionRunner

        self.cfg = load_config(Path(cfg_path))
        self.runner = MotionRunner(self.cfg)

    def step(self, ts: float, frame_bgr: np.ndarray):
        out = self.runner.step(ts, frame_bgr)
        if out is None:
            return 0.0, False, None, None
        return float(out.score), bool(out.pass_frame), out.frame_resized, out.vis_bgr

    def close(self):
        try:
            self.runner.close()
        except Exception:
            pass


class YoloAdapter:
    def __init__(self, cfg_path: str, weights_path: str):
        from ultralytics import YOLO
        import yaml

        self.model = YOLO(weights_path)

        with open(cfg_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f) or {}

        self.imgsz = int(self.cfg.get("imgsz", 416))
        self.conf = float(self.cfg.get("conf", 0.25))
        self.iou = float(self.cfg.get("iou", 0.45))
        self.device = self.cfg.get("device", 0)
        self.verbose = bool(self.cfg.get("verbose", False))

    def detect_persons(self, frame_bgr: np.ndarray):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self.model.predict(
            source=frame_rgb,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=self.verbose,
        )[0]

        out = []
        if res.boxes is None:
            return out

        boxes = res.boxes
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)

        h, w = frame_bgr.shape[:2]

        for (x1, y1, x2, y2), c, cls in zip(xyxy, confs, clss):
            if cls != 0:
                continue

            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            bw = max(1, x2 - x1)
            bh = max(1, y2 - y1)
            area = bw * bh
            area_ratio = area / float(w * h)
            aspect = bh / float(bw)

            if area_ratio < 0.004:
                continue

            if aspect < 0.9 or aspect > 6.0:
                continue

            out.append((float(c), (x1, y1, x2, y2)))

        out.sort(key=lambda x: x[0], reverse=True)
        return out


class PoseLiveAdapter:
    def __init__(self, cfg_path: str):
        from fight.pose.src.pose_adapter import PoseAdapter
        from fight.pose.src.pose_gate import PoseGate
        import yaml

        self.adapter = PoseAdapter(cfg_path)

        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        tcfg = cfg.get("temporal", {})
        self.gate = PoseGate(
            window_size=int(tcfg.get("window_size", 6)),
            need_positive=int(tcfg.get("need_positive", 2)),
            min_mean_score=float(tcfg.get("min_mean_score", 0.30)),
            peak_score_thr=float(tcfg.get("peak_score_thr", 0.54)),
            min_consecutive=int(tcfg.get("min_consecutive", 2)),
        )

    def check(self, roi_bgr: np.ndarray):
        raw = self.adapter.evaluate(roi_bgr, hist_positive=int(sum(self.gate.hist)))
        dec = self.gate.update(raw.score, raw.ok)

        raw.ok = dec.pose_ok
        raw.hist_positive = dec.hist_positive
        return raw

    def reset(self):
        self.gate.reset()


class Stage3Adapter:
    def __init__(self, cfg_path: str):
        import importlib.util
        import yaml
        import torch

        self.torch = torch

        cfgp = Path(cfg_path).resolve()
        with open(cfgp, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f) or {}

        stage3_root = cfgp.parent.parent
        self.stage3_root = stage3_root

        mcfg = self.cfg.get("model", {})
        icfg = self.cfg.get("input", {})

        self.num_classes = int(mcfg.get("num_classes", 2))
        self.device = str(mcfg.get("device", "cuda"))
        self.amp = bool(mcfg.get("amp", False))

        if self.device.startswith("cuda") and not torch.cuda.is_available():
            self.device = "cpu"

        self.clip_len = int(icfg.get("clip_len", 32))
        self.size = int(icfg.get("size", 224))
        self.pad_value = int(icfg.get("pad_value", 114))

        ckpt_rel = str(mcfg.get("ckpt_path", "")).replace("\\", "/")
        if not ckpt_rel:
            raise RuntimeError("stage3.yaml içinde model.ckpt_path yok")

        p = Path(ckpt_rel)
        candidates = [
            (Path.cwd() / p).resolve(),
            (stage3_root / p).resolve(),
            (stage3_root / "weights" / p.name).resolve(),
        ]
        ckpt_path = None
        for c in candidates:
            if c.exists():
                ckpt_path = c
                break
        if ckpt_path is None:
            raise FileNotFoundError(f"ckpt bulunamadı: {ckpt_rel}")

        ml_path = stage3_root / "src" / "model_loader.py"
        spec = importlib.util.spec_from_file_location("stage3_model_loader", str(ml_path))
        mod = importlib.util.module_from_spec(spec)
        assert spec is not None and spec.loader is not None
        spec.loader.exec_module(mod)

        self.model = mod.load_model(str(ckpt_path), device=self.device, num_classes=self.num_classes)
        self.model.eval()

        mean = torch.tensor([0.43216, 0.394666, 0.37645], dtype=torch.float32).view(1, 3, 1, 1, 1)
        std = torch.tensor([0.22803, 0.22145, 0.216989], dtype=torch.float32).view(1, 3, 1, 1, 1)
        self.mean = mean.to(self.device)
        self.std = std.to(self.device)

    def _preprocess(self, clip_bgr_list):
        torch = self.torch

        frames = clip_bgr_list
        T = self.clip_len

        if len(frames) < T:
            if len(frames) == 0:
                raise RuntimeError("clip boş")
            last = frames[-1]
            frames = frames + [last] * (T - len(frames))
        elif len(frames) > T:
            frames = frames[-T:]

        out = []
        for f in frames:
            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            rgb = resize_with_padding(
                rgb,
                out_size=self.size,
                pad_value=self.pad_value,
                interpolation=cv2.INTER_LINEAR,
            )
            out.append(rgb)

        arr = np.stack(out, axis=0)
        x = torch.from_numpy(arr).to(torch.float32) / 255.0
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.unsqueeze(0)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.to(self.device)
        x = (x - self.mean) / self.std
        return x

    def infer(self, clip_bgr_list):
        torch = self.torch
        x = self._preprocess(clip_bgr_list)

        use_amp = self.amp and str(self.device).startswith("cuda")
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = self.model(x)

            if hasattr(logits, "ndim") and logits.ndim == 2 and logits.shape[1] >= 2:
                prob = torch.softmax(logits, dim=1)[:, 1].item()
            else:
                prob = torch.sigmoid(logits).item()

        return float(prob)