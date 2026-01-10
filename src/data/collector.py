from __future__ import annotations

import csv
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import mediapipe as mp


# Config
@dataclass
class DatasetConfig:
    """
    Collector writes CSV in this layout:
    timestamp,user,label,handedness,x0,y0,z0,...,x20,y20,z20

    This matches your train/analyze scripts.
    """
    data_root: str = "data/raw"
    users: Tuple[str, ...] = ("user_valeriy", "user_mom")

    # 8 known + UNKNOWN (UNKNOWN selected by key "0")
    labels_known: Tuple[str, ...] = ("HELP", "STOP", "WATER", "PAIN", "YES", "NO", "CALL", "OK")
    label_unknown: str = "UNKNOWN"

    # burst save every N frames (if enabled)
    burst_every_n_frames: int = 4

    # MediaPipe Hands
    max_num_hands: int = 1
    min_detection_conf: float = 0.6
    min_tracking_conf: float = 0.6

    # UI
    window_name: str = "Gesture2Text - Dataset Collector"


# Paths / CSV
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def csv_path(cfg: DatasetConfig, user: str) -> Path:
    p = Path(cfg.data_root) / user
    ensure_dir(p)
    return p / "samples.csv"


def csv_header() -> List[str]:
    header = ["timestamp", "user", "label", "handedness"]
    for i in range(21):
        header += [f"x{i}", f"y{i}", f"z{i}"]
    return header


def ensure_csv_initialized(cfg: DatasetConfig) -> None:
    for u in cfg.users:
        p = csv_path(cfg, u)
        if not p.exists() or p.stat().st_size == 0:
            with p.open("w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(csv_header())


# MediaPipe helpers
def handedness_str(results) -> str:
    try:
        if results.multi_handedness and len(results.multi_handedness) > 0:
            return results.multi_handedness[0].classification[0].label
    except Exception:
        pass
    return "Unknown"


def landmarks_flat(hand_landmarks) -> List[float]:
    row: List[float] = []
    for lm in hand_landmarks.landmark:
        row.extend([float(lm.x), float(lm.y), float(lm.z)])
    return row


# Overlay
def draw_overlay(
    frame,
    cfg: DatasetConfig,
    user: str,
    label: str,
    count_for_label: int,
    burst_enabled: bool,
) -> None:
    lines = [
        f"USER:  {user}",
        f"LABEL: {label}",
        f"COUNT (this label): {count_for_label}",
        f"MODE: {'BURST' if burst_enabled else 'MANUAL'}",
        "",
        "Keys:",
        "  1-8  -> select label (known gestures)",
        "  0    -> select UNKNOWN",
        "  S    -> save one sample",
        "  SPACE-> toggle burst mode",
        "  U    -> switch user",
        "  R    -> reset counters (in-memory)",
        "  H    -> help (print to console)",
        "  Q    -> quit",
    ]

    y = 30
    for line in lines:
        cv2.putText(frame, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 2, cv2.LINE_AA)
        y += 26


def print_help(cfg: DatasetConfig) -> None:
    print("\n=== Gesture2Text Collector ===")
    print("Users:", cfg.users)
    print("Known labels (1..8):", cfg.labels_known)
    print("UNKNOWN (0):", cfg.label_unknown)
    print("Keys: 1-8 label | 0 unknown | S save | SPACE burst | U user | R reset | Q quit\n")


# Saving
def try_save_sample(
    cfg: DatasetConfig,
    user: str,
    label: str,
    results,
    counts: Dict[Tuple[str, str], int],
) -> bool:
    if not results.multi_hand_landmarks:
        return False

    hand = results.multi_hand_landmarks[0]
    handed = handedness_str(results)

    row = [
        int(time.time() * 1000),
        user,
        label,
        handed,
        *landmarks_flat(hand),
    ]

    p = csv_path(cfg, user)
    with p.open("a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)

    counts[(user, label)] = counts.get((user, label), 0) + 1
    return True


# Main
def main() -> None:
    cfg = DatasetConfig()
    ensure_csv_initialized(cfg)
    print_help(cfg)

    # state
    user_idx = 0
    current_user = cfg.users[user_idx]
    current_label = cfg.labels_known[0]
    burst_enabled = False
    frame_counter = 0
    counts: Dict[Tuple[str, str], int] = {}

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera not found. Check permissions (macOS) and try again.")
        return

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=cfg.max_num_hands,
        model_complexity=1,
        min_detection_confidence=cfg.min_detection_conf,
        min_tracking_confidence=cfg.min_tracking_conf,
    ) as hands:

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # selfie mirror for comfort
            frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

            count_for_label = counts.get((current_user, current_label), 0)
            draw_overlay(frame, cfg, current_user, current_label, count_for_label, burst_enabled)
            cv2.imshow(cfg.window_name, frame)

            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q")):
                break

            if key in (ord("h"), ord("H")):
                print_help(cfg)

            if key in (ord("u"), ord("U")):
                user_idx = (user_idx + 1) % len(cfg.users)
                current_user = cfg.users[user_idx]

            # unknown
            if key == ord("0"):
                current_label = cfg.label_unknown

            # known labels 1..8
            if ord("1") <= key <= ord("8"):
                idx = key - ord("1")
                if idx < len(cfg.labels_known):
                    current_label = cfg.labels_known[idx]

            # toggle burst
            if key == 32:  # SPACE
                burst_enabled = not burst_enabled
                print(f"[INFO] Burst mode: {'ON' if burst_enabled else 'OFF'}")

            # reset counters
            if key in (ord("r"), ord("R")):
                counts.clear()
                print("[INFO] Counters reset (in-memory).")

            # manual save
            if key in (ord("s"), ord("S")):
                if try_save_sample(cfg, current_user, current_label, results, counts):
                    print(f"[SAVE] {current_user} / {current_label} -> {counts[(current_user, current_label)]}")
                else:
                    print("[WARN] No hand detected — nothing saved.")

            # burst save
            frame_counter += 1
            if burst_enabled and (frame_counter % cfg.burst_every_n_frames == 0):
                try_save_sample(cfg, current_user, current_label, results, counts)

    cap.release()
    cv2.destroyAllWindows()
    print("Collector stopped.")


if __name__ == "__main__":
    main()