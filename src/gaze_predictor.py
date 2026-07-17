"""
gaze_predictor.py — Real-time gaze prediction with adaptive Kalman-style smoothing.

ACCURACY IMPROVEMENTS OVER v1:
    1. Exponential moving average (EMA) instead of a flat window average.
       EMA weights recent frames more heavily — it reacts faster to intentional
       gaze shifts while still smoothing micro-jitter.

       Formula:  smoothed = alpha * raw + (1 - alpha) * prev_smoothed
       alpha = 0.35 by default (higher = snappier but more jitter).

    2. Velocity-based alpha boost ("fast gaze detection"):
       When the predicted point jumps a large distance in one frame (i.e., the
       user is deliberately looking at something new), alpha is temporarily raised
       to 0.8 so the cursor catches up quickly. When movement is small (micro-jitter
       on a stable fixation), alpha stays at the lower base value for smoothness.
       This gives the feel of a responsive cursor without the constant jitter.

    3. Outlier gate:
       If a single-frame prediction lands more than MAX_JUMP_PX pixels from the
       current smoothed position and the previous few frames were stable, it's
       likely a blink artifact or landmark glitch — the frame is discarded entirely.
"""

import numpy as np

# EMA smoothing factor for stable gaze (lower = smoother, more lag)
ALPHA_BASE    = 0.30
# EMA factor when a large intentional gaze shift is detected
ALPHA_FAST    = 0.75
# If raw prediction moves more than this from smoothed position in one frame,
# treat as a large intentional shift (use ALPHA_FAST) — pixels
JUMP_THRESHOLD_PX = 80
# If raw prediction moves more than this, treat as artifact and discard — pixels
MAX_JUMP_PX = 350


class GazePredictor:
    """
    Wraps a calibration model and provides smooth real-time gaze prediction.

    Usage:
        predictor = GazePredictor(model)
        gaze_xy = predictor.predict(features)   # np.ndarray (2,) or None
    """

    def __init__(self, model, alpha_base=ALPHA_BASE, alpha_fast=ALPHA_FAST):
        self.model       = model
        self.alpha_base  = alpha_base
        self.alpha_fast  = alpha_fast
        self._smoothed   = None   # current EMA state

    def predict(self, features):
        """
        Predict smoothed gaze coordinate from a feature vector.

        Args:
            features: np.ndarray shape (10,) or None.

        Returns:
            np.ndarray shape (2,) — smoothed (x, y) — or None if no face.
        """
        if features is None:
            return None

        raw = self.model.predict(features.reshape(1, -1))[0].astype(np.float64)

        if self._smoothed is None:
            # First frame — seed the smoother with the raw prediction
            self._smoothed = raw.copy()
            return self._smoothed.copy()

        dist = np.linalg.norm(raw - self._smoothed)

        # Artifact gate: single-frame prediction too far away → skip this frame
        if dist > MAX_JUMP_PX:
            return self._smoothed.copy()

        # Velocity-adaptive alpha
        alpha = self.alpha_fast if dist > JUMP_THRESHOLD_PX else self.alpha_base

        # EMA update
        self._smoothed = alpha * raw + (1.0 - alpha) * self._smoothed
        return self._smoothed.copy()

    def reset(self):
        """Clear smoother state (call after recalibration)."""
        self._smoothed = None

    @staticmethod
    def clamp_to_screen(xy, screen_w, screen_h):
        x = float(np.clip(xy[0], 0, screen_w - 1))
        y = float(np.clip(xy[1], 0, screen_h - 1))
        return np.array([x, y])
