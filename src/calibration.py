"""
calibration.py — High-accuracy 16-point calibration with per-frame sample retention.

ACCURACY IMPROVEMENTS OVER v1:
    1. Keeps ALL raw frame samples (not just the per-point mean).
       9 points × ~45 frames = ~405 training samples vs the previous 9.
       More samples let the MLP learn the nonlinear iris-to-screen mapping
       much more reliably.

    2. 16-point grid (4×4) instead of 9-point (3×3).
       More anchor points = better coverage of the screen corners and edges
       where prediction error tends to be highest.

    3. Discards the first 40% of frames at each point (transition/settling frames).
       Only the stable "locked-on" portion of each dwell window is used for
       training, reducing noise from eye movement while gaze settles onto the dot.

    4. Outlier rejection: per-point, samples more than 2 std-deviations from the
       mean feature vector are dropped before averaging, removing blink artifacts.

    5. GradientBoostingRegressor instead of MLPRegressor.
       GBR is an ensemble of decision trees; it generalizes better with moderate
       sample counts (~400), doesn't require careful learning-rate tuning, and is
       naturally resistant to overfitting without explicit regularization.
       We use a MultiOutputRegressor wrapper for the (x, y) two-output case.

    6. Calibration quality score is shown after fitting so you know if you
       should redo it.
"""

import os
import pickle
import time

import cv2
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from gaze_tracker import GazeTracker

# ── Paths ──────────────────────────────────────────────────────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CALIBRATION_DIR  = os.path.join(_THIS_DIR, "..", "calibration_data")
CALIBRATION_FILE = os.path.join(CALIBRATION_DIR, "calibration_model.pkl")

# ── Parameters ─────────────────────────────────────────────────────────────────
DWELL_SECONDS   = 2.5   # time to show each calibration point
LEAD_IN_SECONDS = 0.8   # pause before sample collection (let gaze settle)
SKIP_FRAC       = 0.40  # discard first 40% of each dwell window
MARGIN_FRAC     = 0.08  # keep dots away from screen edges

DOT_RADIUS   = 16
RING_RADIUS  = 32
PULSE_PERIOD = 1.0  # seconds for pulse animation

# Colors (BGR)
COL_BG        = (12, 12, 18)
COL_DOT       = (0, 230, 120)
COL_DOT_DONE  = (40, 80, 40)
COL_RING      = (0, 180, 80)
COL_PROGRESS  = (0, 210, 255)
COL_WHITE     = (255, 255, 255)
COL_DIM       = (80, 80, 80)
COL_TEXT      = (200, 200, 200)
COL_SUBTITLE  = (100, 140, 100)


def _make_grid_points(w, h):
    """4×4 grid of calibration points with margin from edges."""
    mx = int(w * MARGIN_FRAC)
    my = int(h * MARGIN_FRAC)
    cols = [mx, w // 3, 2 * w // 3, w - mx]
    rows = [my, h // 3, 2 * h // 3, h - my]
    return [(x, y) for y in rows for x in cols]


def run_calibration(tracker: GazeTracker, screen_w: int, screen_h: int) -> bool:
    """
    Run the interactive 16-point calibration.

    Collects raw per-frame gaze features for each screen point,
    fits a gradient boosting regressor, saves to disk.

    Returns True on success, False if aborted.
    """
    os.makedirs(CALIBRATION_DIR, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam for calibration.")
        return False
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    points = _make_grid_points(screen_w, screen_h)
    all_features = []
    all_targets  = []

    cv2.namedWindow("GazeType — Calibration", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("GazeType — Calibration",
                          cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    aborted = False

    for point_idx, (px, py) in enumerate(points):

        # ── Lead-in pause ──────────────────────────────────────────────────────
        lead_end = time.time() + LEAD_IN_SECONDS
        while time.time() < lead_end:
            canvas = _blank(screen_h, screen_w)
            _draw_scene(canvas, points, point_idx, (px, py),
                        screen_w, screen_h, 0.0)
            cv2.imshow("GazeType — Calibration", canvas)
            if cv2.waitKey(1) & 0xFF == 27:
                aborted = True
                break
        if aborted:
            break

        # ── Sample collection ──────────────────────────────────────────────────
        collect_start = time.time()
        point_raw = []

        while True:
            elapsed = time.time() - collect_start
            progress = min(elapsed / DWELL_SECONDS, 1.0)

            ret, frame = cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                landmarks = tracker.process_frame(frame)
                features  = tracker.get_features_from_landmarks(landmarks)

                if features is not None:
                    # Only collect from the stable tail of the dwell window
                    if elapsed >= DWELL_SECONDS * SKIP_FRAC:
                        point_raw.append(features)

                    # Record head pose from the centre of the first point
                    if point_idx == 0 and elapsed > LEAD_IN_SECONDS / 2:
                        tracker.record_calibration_pose(landmarks)

            canvas = _blank(screen_h, screen_w)
            _draw_scene(canvas, points, point_idx, (px, py),
                        screen_w, screen_h, progress)
            cv2.imshow("GazeType — Calibration", canvas)

            if cv2.waitKey(1) & 0xFF == 27:
                aborted = True
                break
            if elapsed >= DWELL_SECONDS:
                break

        if aborted:
            break

        if len(point_raw) < 3:
            print(f"WARNING: sparse samples at point {point_idx + 1}")

        # Outlier rejection: drop samples > 2σ from per-point mean
        if len(point_raw) >= 4:
            arr = np.array(point_raw)
            mean  = arr.mean(axis=0)
            std   = arr.std(axis=0) + 1e-8
            zscores = np.abs((arr - mean) / std).mean(axis=1)
            clean = arr[zscores <= 2.0]
            point_raw = clean.tolist() if len(clean) >= 2 else point_raw

        for feat in point_raw:
            all_features.append(feat)
            all_targets.append([float(px), float(py)])

    cap.release()
    cv2.destroyAllWindows()

    if aborted:
        print("Calibration aborted.")
        return False

    if len(all_features) < 20:
        print("ERROR: Not enough calibration data. Try again.")
        return False

    # ── Fit model ─────────────────────────────────────────────────────────────
    X = np.array(all_features)
    y = np.array(all_targets)

    print(f"Fitting model on {len(X)} samples across {len(points)} points...")

    base = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.85,
        random_state=42,
    )
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("gbr",    MultiOutputRegressor(base)),
    ])
    model.fit(X, y)

    # Calibration quality: mean training error in pixels (lower is better)
    y_pred = model.predict(X)
    err = np.linalg.norm(y_pred - y, axis=1).mean()
    print(f"Calibration training error: {err:.1f} px  "
          f"({'good' if err < 60 else 'ok' if err < 120 else 'consider redoing'})")

    with open(CALIBRATION_FILE, "wb") as f:
        pickle.dump(model, f)
    print(f"Saved to {CALIBRATION_FILE}")
    return True


def load_calibration():
    if not os.path.exists(CALIBRATION_FILE):
        return None
    with open(CALIBRATION_FILE, "rb") as f:
        return pickle.load(f)


def calibration_exists():
    return os.path.exists(CALIBRATION_FILE)


# ── Drawing helpers ────────────────────────────────────────────────────────────

def _blank(h, w):
    return np.full((h, w, 3), COL_BG, dtype=np.uint8)


def _draw_scene(canvas, points, current_idx, current_pt,
                sw, sh, progress):
    px, py = current_pt
    total  = len(points)
    now    = time.time()

    # ── Title ─────────────────────────────────────────────────────────────────
    _ctext(canvas, "G A Z E T Y P E", sh // 2 - 120, sw,
           scale=1.1, color=(0, 220, 120), thickness=2)
    _ctext(canvas, f"Look at the dot and hold still",
           sh // 2 - 70, sw, scale=0.65, color=COL_TEXT)
    _ctext(canvas, f"Point {current_idx + 1} of {total}",
           sh // 2 - 38, sw, scale=0.55, color=COL_SUBTITLE)
    _ctext(canvas, "ESC to abort",
           sh - 28, sw, scale=0.45, color=(60, 60, 60))

    # ── Progress bar (bottom of screen) ───────────────────────────────────────
    bar_h = 6
    bar_y = sh - bar_h - 2
    bar_w = int(sw * (current_idx / total))
    cv2.rectangle(canvas, (0, bar_y), (sw, sh), (20, 20, 20), -1)
    cv2.rectangle(canvas, (0, bar_y), (bar_w, sh), (0, 160, 80), -1)

    # ── Completed dots (small) ────────────────────────────────────────────────
    for i, (ox, oy) in enumerate(points):
        if i < current_idx:
            cv2.circle(canvas, (ox, oy), 5, COL_DOT_DONE, -1)

    # ── Pulsing ring + progress arc ────────────────────────────────────────────
    pulse = 0.7 + 0.3 * abs(np.sin(now * np.pi / PULSE_PERIOD))
    ring_r = int(RING_RADIUS * (1.0 + 0.15 * pulse))
    cv2.circle(canvas, (px, py), ring_r, COL_DIM, 1)

    if progress > 0:
        angle = int(360 * progress)
        cv2.ellipse(canvas, (px, py), (RING_RADIUS, RING_RADIUS),
                    -90, 0, angle, COL_PROGRESS, 3)

    # ── Main dot ──────────────────────────────────────────────────────────────
    dot_color = (0, 255, 200) if progress >= 1.0 else COL_DOT
    cv2.circle(canvas, (px, py), DOT_RADIUS, dot_color, -1)
    cv2.circle(canvas, (px, py), 4,          COL_WHITE,  -1)   # precision aiming


def _ctext(img, text, y, width, scale=0.7,
           color=(220, 220, 220), thickness=1):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, _), _ = cv2.getTextSize(text, font, scale, thickness)
    x = (width - tw) // 2
    cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)
