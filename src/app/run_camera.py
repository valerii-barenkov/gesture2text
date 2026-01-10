# src/app/run_camera.py
from __future__ import annotations

import os
import platform
import subprocess
import warnings
from pathlib import Path
from typing import Optional

import cv2
import joblib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ml.features import extract_features
from vision.hand_tracker import HandTracker


# меньше мусора в консоли (MediaPipe/TFLite)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 0/1/2/3 (2 = warnings+, 3 = errors only)
warnings.filterwarnings("ignore", message=r".*GetPrototype\(\) is deprecated.*")


# Paths / bundle
def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def pick_model_path(root: Path) -> Path:
    best = root / "src" / "data" / "models" / "gesture2text_combined_frozen_best.joblib"
    default = root / "src" / "data" / "models" / "gesture2text_combined_frozen.joblib"
    return best if best.exists() else default


def load_bundle(model_path: Path):
    obj = joblib.load(model_path)

    if isinstance(obj, dict):
        if "model" not in obj:
            raise ValueError("Model bundle is dict, but no key 'model'.")
        return obj["model"], obj.get("classes"), obj.get("n_features")

    return obj, None, None


# Helpers
def now_ms() -> int:
    return int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)


def landmarks_to_row(landmarks_norm: np.ndarray, handedness: Optional[str]) -> dict:
    """x0,y0,z0 ... x20,y20,z20 + handedness"""
    row: dict = {}
    for i, (x, y, z) in enumerate(landmarks_norm):
        row[f"x{i}"] = float(x)
        row[f"y{i}"] = float(y)
        row[f"z{i}"] = float(z)
    row["handedness"] = handedness or "Right"
    return row


# TTS
try:
    import pyttsx3  # optional; Win/Linux
except Exception:
    pyttsx3 = None

_SYSTEM = platform.system()
_TTS_ENGINE = None


def speak(text: str) -> bool:
    global _TTS_ENGINE
    if not text:
        return False

    if _SYSTEM == "Darwin":
        try:
            subprocess.Popen(["say", text])
            return True
        except Exception:
            return False

    if pyttsx3 is None:
        return False

    try:
        if _TTS_ENGINE is None:
            _TTS_ENGINE = pyttsx3.init()
            _TTS_ENGINE.setProperty("rate", 165)
        _TTS_ENGINE.say(text)
        _TTS_ENGINE.runAndWait()
        return True
    except Exception:
        return False


# UI (PIL unicode-safe)
def _font_candidates() -> list[Path]:
    system = platform.system()
    candidates: list[Path] = []

    if system == "Darwin":
        candidates += [
            Path("/System/Library/Fonts/SFNS.ttf"),
            Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
            Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
        ]
    elif system == "Windows":
        win = Path(os.environ.get("WINDIR", "C:/Windows"))
        candidates += [win / "Fonts" / "segoeui.ttf", win / "Fonts" / "arial.ttf"]
    else:
        candidates += [
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
            Path("/usr/share/fonts/truetype/freefont/FreeSans.ttf"),
        ]

    return candidates


def load_fonts(size_base: int = 28):
    base = None
    for p in _font_candidates():
        if p.exists():
            try:
                base = ImageFont.truetype(str(p), size_base)
                break
            except Exception:
                pass

    if base is None:
        base = ImageFont.load_default()

    # HUD smaller / bottom bigger
    try:
        hud = ImageFont.truetype(getattr(base, "path", ""), int(size_base * 0.80)) if hasattr(base, "path") else base
    except Exception:
        hud = base
    try:
        bottom = ImageFont.truetype(getattr(base, "path", ""), int(size_base * 1.65)) if hasattr(base, "path") else base
    except Exception:
        bottom = base

    return hud, bottom


def _measure(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    x0, y0, x1, y1 = draw.textbbox((0, 0), text, font=font)
    return x1 - x0, y1 - y0


def draw_hud(frame: np.ndarray, lines: list[str], font: ImageFont.ImageFont) -> np.ndarray:
    if not lines:
        return frame

    h, w = frame.shape[:2]
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    pad = 12
    gap = 6

    widths, heights = zip(*[_measure(draw, s, font) for s in lines])
    box_w = max(widths) + pad * 2
    box_h = sum(heights) + pad * 2 + gap * (len(lines) - 1)

    x0, y0 = 15, 15
    x1, y1 = min(w - 10, x0 + box_w), y0 + box_h

    draw.rectangle([x0, y0, x1, y1], fill=(0, 0, 0))

    y = y0 + pad
    for s, th in zip(lines, heights):
        draw.text((x0 + pad, y), s, font=font, fill=(255, 255, 255))
        y += th + gap

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def draw_bottom_label(frame: np.ndarray, text: str, font: ImageFont.ImageFont, text_rgb=(0, 255, 0)) -> np.ndarray:
    if not text:
        return frame

    h, w = frame.shape[:2]
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    tw, th = _measure(draw, text, font)
    x = (w - tw) // 2
    y = h - th - 30

    pad = 14
    draw.rectangle([x - pad, y - pad, x + tw + pad, y + th + pad], fill=(0, 0, 0))
    draw.text((x, y), text, font=font, fill=text_rgb)

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def draw_sound_bars(frame: np.ndarray, enabled: bool, speaking: bool, t_ms: int, proba: Optional[float]) -> np.ndarray:
    _, w = frame.shape[:2]

    bar_w, gap, bars, max_h = 6, 3, 6, 28
    base_x = w - (bars * bar_w + (bars - 1) * gap) - 18
    base_y = 20

    color = (0, 255, 0) if enabled else (220, 220, 220)

    phase = (t_ms % 600) / 600.0
    pulse = 0.5 + 0.5 * np.sin(2 * np.pi * phase)

    if enabled and speaking:
        activity = 0.35 + 0.65 * float(pulse)
    elif enabled and proba is not None:
        activity = 0.20 + 0.35 * float(np.clip(proba, 0.0, 1.0))
    else:
        activity = 0.15

    for i in range(bars):
        x0 = base_x + i * (bar_w + gap)
        wave = 0.65 + 0.35 * np.sin(2 * np.pi * (phase + i * 0.12))
        bh = int(max_h * activity * wave)
        bh = max(6, min(max_h, bh))
        y0 = base_y + (max_h - bh)
        y1 = base_y + max_h
        cv2.rectangle(frame, (x0, y0), (x0 + bar_w, y1), color, -1)

    return frame


# Main
def main() -> None:
    root = project_root()
    model_path = pick_model_path(root)

    if not model_path.exists():
        print(f"[ERROR] Model not found: {model_path}")
        print("Train first: PYTHONPATH=src python src/ml/train.py --dataset combined")
        return

    model, classes, n_features = load_bundle(model_path)
    print(f"[OK] Loaded model: {model_path}")
    if n_features is not None:
        print(f"[INFO] Model expects n_features={n_features}")

    hud_font, bottom_font = load_fonts(size_base=28)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera not available. Check macOS permissions.")
        return

    tracker = HandTracker(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    print("Camera started. Press Q to quit.")

    # UX
    mode_debug = False
    speech_enabled = False
    show_proba_in_simple = True

    # gate
    proba_thr = 0.78
    margin_thr = 0.16
    stable_frames = 4

    # speech
    speech_cooldown_ms = 1500
    speech_icon_ms = 450
    last_spoken = 0
    speaking_until = 0

    # stable state
    stable_label: Optional[str] = None
    stable_count = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            result = tracker.process(frame)

            label_text = ""
            label_color = (0, 255, 0)
            top1 = top2 = margin = None
            overlay_proba: Optional[float] = None
            confident = False
            confirmed = False

            if result.landmarks_norm is not None:
                frame = tracker.draw(frame, result.landmarks_norm)

                row = landmarks_to_row(result.landmarks_norm, result.handedness)
                feat = extract_features(row)

                if feat is not None:
                    X = np.asarray(feat, dtype=float).reshape(1, -1)
                    if not np.isfinite(X).all():
                        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

                    if hasattr(model, "predict_proba"):
                        proba = model.predict_proba(X)[0].astype(float)
                        idx = np.argsort(proba)
                        i1 = int(idx[-1])
                        i2 = int(idx[-2]) if len(idx) > 1 else i1

                        top1 = float(proba[i1])
                        top2 = float(proba[i2]) if len(idx) > 1 else 0.0
                        margin = float(top1 - top2)
                        overlay_proba = top1

                        pred = str(classes[i1]) if classes is not None else str(i1)
                        confident = (top1 >= proba_thr) and (margin >= margin_thr)
                    else:
                        pred = str(model.predict(X)[0])
                        confident = True

                    if confident:
                        if pred == stable_label:
                            stable_count += 1
                        else:
                            stable_label = pred
                            stable_count = 1
                    else:
                        stable_label = None
                        stable_count = 0

                    confirmed = stable_label is not None and stable_count >= stable_frames

                    if confirmed and stable_label is not None:
                        if overlay_proba is not None and (mode_debug or show_proba_in_simple):
                            label_text = f"{stable_label} ({overlay_proba:.2f})"
                        else:
                            label_text = stable_label

                        if speech_enabled:
                            t = now_ms()
                            if t - last_spoken >= speech_cooldown_ms:
                                if speak(stable_label):
                                    speaking_until = t + speech_icon_ms
                                last_spoken = t
                    else:
                        if result.landmarks_norm is not None and not confident:
                            label_text = "NO GESTURE"
                            label_color = (220, 220, 220)

            controls = "Q: Quit  |  S: Sound  |  M: Mode"
            speech_status = "ON" if speech_enabled else "OFF"

            if mode_debug:
                if top1 is None:
                    dbg = "p1/p2/margin: n/a (no predict_proba)"
                    gate = "gate: n/a"
                else:
                    dbg = f"p1: {top1:.2f}  p2: {top2:.2f}  margin: {margin:.2f}"
                    gate = f"gate: {'PASS' if confident else 'REJECT'}  |  hold: {stable_count}/{stable_frames}"

                hud = [
                    "Gesture2Text — Debug",
                    controls,
                    f"Sound: {speech_status}  |  Thr: {proba_thr:.2f}  |  Margin: {margin_thr:.2f}  |  Stable: {stable_frames}",
                    dbg,
                    gate,
                    f"LOCK: {'YES' if confirmed else 'NO'}",
                ]
            else:
                hud = ["Gesture2Text", controls, f"Sound: {speech_status}"]

            frame = draw_hud(frame, hud, font=hud_font)

            t = now_ms()
            speaking = t <= speaking_until
            frame = draw_sound_bars(frame, speech_enabled, speaking, t, overlay_proba)

            frame = draw_bottom_label(frame, label_text, font=bottom_font, text_rgb=label_color)

            cv2.imshow("Gesture2Text Live", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                break
            if key in (ord("s"), ord("S")):
                speech_enabled = not speech_enabled
                print(f"[INFO] Sound: {'ON' if speech_enabled else 'OFF'}")
            if key in (ord("m"), ord("M")):
                mode_debug = not mode_debug
                print(f"[INFO] Mode: {'DEBUG' if mode_debug else 'SIMPLE'}")

    finally:
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()