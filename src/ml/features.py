# src/ml/features.py
import numpy as np

# MediaPipe landmark indices
WRIST = 0
MIDDLE_FINGER_TIP = 12

EPS = 1e-6


def extract_features(row) -> np.ndarray | None:
    """
    Convert a single CSV row (one frame) into a feature vector of shape (63,).

    Steps:
    - read 21 hand landmarks (x, y, z)
    - center coordinates at the wrist
    - normalize by hand scale (middle finger tip distance)
    - mirror left hand to right-hand coordinate system
    - flatten to 1D vector

    Returns:
        np.ndarray (63,) on success
        None if the frame is invalid
    """

    # read landmark coordinates
    coords = np.empty((21, 3), dtype=float)
    try:
        for i in range(21):
            coords[i, 0] = float(row[f"x{i}"])
            coords[i, 1] = float(row[f"y{i}"])
            coords[i, 2] = float(row[f"z{i}"])
    except Exception:
        return None

    if not np.isfinite(coords).all():
        return None

    # center at wrist
    coords -= coords[WRIST]

    # normalize by hand size
    scale = float(np.linalg.norm(coords[MIDDLE_FINGER_TIP]))
    if not np.isfinite(scale) or scale < EPS:
        return None

    coords /= scale

    # mirror left hand to right-hand system
    handedness = None
    try:
        handedness = row.get("handedness", None)
    except Exception:
        pass

    if isinstance(handedness, str) and handedness.lower().startswith("l"):
        coords[:, 0] *= -1.0

    feat = coords.reshape(-1)

    if not np.isfinite(feat).all():
        return None

    return feat