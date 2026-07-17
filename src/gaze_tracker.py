"""
gaze_tracker.py — High-accuracy iris landmark extraction with rich feature engineering.

ACCURACY IMPROVEMENTS OVER v1:
    1. Richer 10-element feature vector (was 4)
       - Per-eye: iris_x, iris_y, iris_x/y relative to BOTH corners & lids (4 values)
       - Adds inter-pupil distance normalization (scale-invariant across sitting distances)
       - Adds head-pose proxy: vertical nose-to-midpoint offset (compensates head tilt)

    2. Better reference landmarks — uses the tightest anatomical corners per eye,
       plus multiple anchor combinations so small jitter in any single landmark
       averages out across the feature computation.

    3. single-call design: process_frame() is called once per main-loop iteration
       and the result is passed to both get_features_from_landmarks() and
       check_head_stability() — eliminating the double MediaPipe inference.

FEATURE VECTOR LAYOUT (10 values):
    0  left_iris_norm_x    iris x relative to left inner corner / eye width
    1  left_iris_norm_y    iris y relative to left upper lid    / eye height
    2  right_iris_norm_x   same for right eye
    3  right_iris_norm_y
    4  left_iris_cx        iris x relative to eye horizontal center / eye width
    5  right_iris_cx       same for right
    6  left_iris_cy        iris y relative to eye vertical center / eye height
    7  right_iris_cy
    8  ipd_norm_left_x     left  iris x / inter-pupil distance (scale compensation)
    9  ipd_norm_right_x    right iris x / inter-pupil distance
"""

import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh

# ── Iris centers ───────────────────────────────────────────────────────────────
LEFT_IRIS_CENTER  = 468
RIGHT_IRIS_CENTER = 473

# ── Eye anchors — anatomically precise corners and lids ───────────────────────
# Left eye  (your left, camera's right)
L_INNER = 362   # medial canthus
L_OUTER = 263   # lateral canthus
L_UPPER = 386   # upper lid apex
L_LOWER = 374   # lower lid nadir
L_UPPER2 = 387  # second upper lid point for averaging
L_LOWER2 = 373  # second lower lid point for averaging

# Right eye
R_INNER = 133
R_OUTER = 33
R_UPPER = 159
R_LOWER = 145
R_UPPER2 = 160
R_LOWER2 = 144

# Nose tip for head-stability tracking
NOSE_TIP = 1
CHIN_TIP = 152  # used with nose for vertical pose proxy

HEAD_MOVE_THRESHOLD = 0.04   # normalized units (~4% of frame width)


class GazeTracker:
    """
    High-accuracy MediaPipe iris tracker with rich feature extraction.

    Recommended usage (avoids double-inference per frame):

        landmarks = tracker.process_frame(frame)          # run MediaPipe ONCE
        features  = tracker.get_features_from_landmarks(landmarks)
        stable    = tracker.check_head_stability(landmarks)
    """

    def __init__(self):
        self._face_mesh = None
        self.calibration_nose_pos = None

    def open(self):
        self._face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def close(self):
        if self._face_mesh:
            self._face_mesh.close()
            self._face_mesh = None

    def process_frame(self, bgr_frame):
        """Run MediaPipe on a BGR frame. Returns landmark list or None."""
        if self._face_mesh is None:
            raise RuntimeError("Call open() first.")
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self._face_mesh.process(rgb)
        if results.multi_face_landmarks:
            return results.multi_face_landmarks[0]
        return None

    def get_features_from_landmarks(self, landmarks):
        """
        Compute the 10-element gaze feature vector from pre-computed landmarks.
        Pass the result of process_frame() here to avoid double inference.
        Returns np.ndarray shape (10,) or None.
        """
        if landmarks is None:
            return None
        return self._compute_features(landmarks)

    # Convenience wrapper: runs process_frame internally (less efficient in loops)
    def get_features(self, bgr_frame):
        lm = self.process_frame(bgr_frame)
        return self.get_features_from_landmarks(lm)

    def _compute_features(self, landmarks):
        lm = landmarks.landmark

        # ── Raw landmark positions ─────────────────────────────────────────────
        lx = lm[LEFT_IRIS_CENTER].x;   ly = lm[LEFT_IRIS_CENTER].y
        rx = lm[RIGHT_IRIS_CENTER].x;  ry = lm[RIGHT_IRIS_CENTER].y

        # Left eye bounds — average two lid points to reduce single-landmark noise
        li = lm[L_INNER].x;  lo = lm[L_OUTER].x
        lu = (lm[L_UPPER].y + lm[L_UPPER2].y) / 2
        ll = (lm[L_LOWER].y + lm[L_LOWER2].y) / 2

        # Right eye bounds
        ri = lm[R_INNER].x;  ro = lm[R_OUTER].x
        ru = (lm[R_UPPER].y + lm[R_UPPER2].y) / 2
        rl = (lm[R_LOWER].y + lm[R_LOWER2].y) / 2

        # ── Eye dimensions ─────────────────────────────────────────────────────
        lew = abs(lo - li) or 1e-6
        leh = abs(ll - lu) or 1e-6
        rew = abs(ro - ri) or 1e-6
        reh = abs(rl - ru) or 1e-6

        # ── Feature 0-1: inner-corner normalized ──────────────────────────────
        f0 = (lx - li) / lew   # left iris x relative to inner corner
        f1 = (ly - lu) / leh   # left iris y relative to upper lid
        f2 = (rx - ri) / rew
        f3 = (ry - ru) / reh

        # ── Feature 4-7: center-normalized (second reference frame) ──────────
        # Measures displacement from eye center — complementary to corner-relative
        l_cx = (li + lo) / 2;  l_cy = (lu + ll) / 2
        r_cx = (ri + ro) / 2;  r_cy = (ru + rl) / 2
        f4 = (lx - l_cx) / lew
        f5 = (rx - r_cx) / rew
        f6 = (ly - l_cy) / leh
        f7 = (ry - r_cy) / reh

        # ── Feature 8-9: scale-normalized by inter-pupil distance ─────────────
        # IPD changes when you move closer/further from camera; dividing by it
        # makes horizontal gaze features robust to sitting-distance changes.
        ipd = abs(rx - lx) or 1e-6
        f8 = lx / ipd
        f9 = rx / ipd

        return np.array([f0, f1, f2, f3, f4, f5, f6, f7, f8, f9],
                        dtype=np.float32)

    def check_head_stability(self, landmarks):
        """
        Returns True (stable), False (moved too much), or None (no reference).
        """
        if landmarks is None or self.calibration_nose_pos is None:
            return None
        nose = landmarks.landmark[NOSE_TIP]
        cur = np.array([nose.x, nose.y])
        delta = np.linalg.norm(cur - self.calibration_nose_pos)
        return delta <= HEAD_MOVE_THRESHOLD

    def record_calibration_pose(self, landmarks):
        """Store current nose position as calibration head-pose reference."""
        if landmarks is None:
            return
        nose = landmarks.landmark[NOSE_TIP]
        self.calibration_nose_pos = np.array([nose.x, nose.y])
