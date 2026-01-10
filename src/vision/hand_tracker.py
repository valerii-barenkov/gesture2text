# src/vision/hand_tracker.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.framework.formats import landmark_pb2


@dataclass
class HandResult:
    """Result of hand detection for a single frame"""
    landmarks_px: Optional[np.ndarray]     # (21, 2) pixel coordinates
    landmarks_norm: Optional[np.ndarray]   # (21, 3) normalized coordinates
    handedness: Optional[str]              # 'Left' / 'Right'
    confidence: Optional[float]            # handedness confidence


class HandTracker:
    """Lightweight wrapper around MediaPipe Hands"""

    def __init__(
        self,
        static_image_mode: bool = False,
        max_num_hands: int = 1,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.6,
        min_tracking_confidence: float = 0.6,
    ) -> None:
        self._mp_hands = mp.solutions.hands
        self._mp_draw = mp.solutions.drawing_utils

        # MediaPipe Hands instance
        self._hands = self._mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process(self, frame_bgr: np.ndarray) -> HandResult:
        """Detect a hand and return landmarks for one frame"""
        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self._hands.process(frame_rgb)

        if not result.multi_hand_landmarks:
            return HandResult(None, None, None, None)

        # We use only the first hand (max_num_hands = 1)
        hand = result.multi_hand_landmarks[0]

        # Normalized landmarks (0..1)
        landmarks_norm = np.array(
            [[lm.x, lm.y, lm.z] for lm in hand.landmark],
            dtype=np.float32,
        )

        # Pixel coordinates
        landmarks_px = np.array(
            [[lm.x * w, lm.y * h] for lm in hand.landmark],
            dtype=np.float32,
        )

        handedness = None
        confidence = None
        if result.multi_handedness:
            info = result.multi_handedness[0].classification[0]
            handedness = info.label
            confidence = float(info.score)

        return HandResult(
            landmarks_px=landmarks_px,
            landmarks_norm=landmarks_norm,
            handedness=handedness,
            confidence=confidence,
        )

    def draw(self, frame_bgr: np.ndarray, landmarks_norm: np.ndarray) -> np.ndarray:
        """Draw hand landmarks on a frame"""
        landmark_list = landmark_pb2.NormalizedLandmarkList()

        for x, y, *rest in landmarks_norm:
            z = rest[0] if rest else 0.0
            landmark_list.landmark.append(
                landmark_pb2.NormalizedLandmark(
                    x=float(x),
                    y=float(y),
                    z=float(z),
                )
            )

        out = frame_bgr.copy()
        self._mp_draw.draw_landmarks(
            out,
            landmark_list,
            self._mp_hands.HAND_CONNECTIONS,
        )
        return out

    def close(self) -> None:
        """Release MediaPipe resources"""
        self._hands.close()