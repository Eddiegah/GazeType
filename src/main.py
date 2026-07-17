"""
main.py — GazeType application entry point.

Startup → calibration decision → keyboard loop.

Runtime keys:
    R      recalibrate without restarting
    Q/ESC  quit
"""

import os
import sys
import time
import math

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gaze_tracker   import GazeTracker
from calibration    import run_calibration, load_calibration, calibration_exists
from gaze_predictor import GazePredictor
from keyboard_ui    import KeyboardUI, FONT

WINDOW_NAME = "GazeType"

# ── Palette (shared with screens) ─────────────────────────────────────────────
C_BG     = (10,  10,  16)
C_PANEL  = (18,  18,  28)
C_ACCENT = (0,  200, 120)
C_ACCT2  = (0,  160, 255)
C_TEXT   = (200, 205, 215)
C_DIM    = (70,  75,  85)
C_WHITE  = (240, 245, 255)
C_WARN   = (60, 100, 220)
C_TITLE  = (0,  230, 140)
C_GRAD1  = (0,  180, 120)  # gradient start
C_GRAD2  = (0,  130, 220)  # gradient end


# ── Screen helpers ─────────────────────────────────────────────────────────────

def get_screen_size():
    try:
        tmp = "_gt_probe"
        cv2.namedWindow(tmp, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(tmp, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.waitKey(200)
        _, _, w, h = cv2.getWindowImageRect(tmp)
        cv2.destroyWindow(tmp)
        cv2.waitKey(100)
        if w > 400 and h > 300:
            return w, h
    except Exception:
        pass
    return 1920, 1080


def _ensure_fullscreen():
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
                          cv2.WINDOW_FULLSCREEN)


# ── Animated splash / prompt screens ─────────────────────────────────────────

class _AnimState:
    """Tiny namespace for animation timing."""
    start = time.time()

    @staticmethod
    def t():
        return time.time() - _AnimState.start


def _draw_bg(canvas, sw, sh):
    """Animated grid background."""
    t = _AnimState.t()
    # Draw subtle grid lines
    step = 60
    offset = int(t * 20) % step
    for x in range(-step, sw + step, step):
        alpha = 0.04
        xi = x + offset
        if 0 <= xi < sw:
            cv2.line(canvas, (xi, 0), (xi, sh), (30, 35, 50), 1)
    for y in range(0, sh, step):
        cv2.line(canvas, (0, y), (sw, y), (30, 35, 50), 1)


def _draw_logo(canvas, sw, sh, cy_offset=0):
    """Draw the GazeType title with gradient-style lettering."""
    t = _AnimState.t()
    title = "GAZETYPE"
    scale = 3.2
    thick = 4
    (tw, th), _ = cv2.getTextSize(title, FONT, scale, thick)
    tx = (sw - tw) // 2
    ty = sh // 2 - 120 + cy_offset

    # Shadow
    cv2.putText(canvas, title, (tx + 3, ty + 3), FONT, scale,
                (0, 30, 20), thick + 2, cv2.LINE_AA)
    # Main text (accent color)
    cv2.putText(canvas, title, (tx, ty), FONT, scale, C_TITLE, thick, cv2.LINE_AA)

    # Tagline
    tag = "Hands-Free Typing  ·  Eyes Only"
    stag = 0.6
    (tagw, _), _ = cv2.getTextSize(tag, FONT, stag, 1)
    cv2.putText(canvas, tag, ((sw - tagw) // 2, ty + th + 10),
                FONT, stag, C_DIM, 1, cv2.LINE_AA)

    # Animated underline
    ul_w = int(tw * min(t / 0.8, 1.0))
    cv2.line(canvas, (tx, ty + 8), (tx + ul_w, ty + 8), C_ACCENT, 3)


def _draw_option_box(canvas, x, y, w, h, key_char, label, desc,
                     highlighted=False, anim_t=0.0):
    """Draw a styled option button with key shortcut, label, and description."""
    # Background
    alpha = 0.7 if highlighted else 0.4
    overlay = canvas.copy()
    r = 10
    col = (35, 50, 35) if highlighted else (22, 22, 34)
    cv2.rectangle(overlay, (x + r, y), (x + w - r, y + h), col, -1)
    cv2.rectangle(overlay, (x, y + r), (x + w, y + h - r), col, -1)
    for cx_, cy_ in [(x+r, y+r), (x+w-r, y+r), (x+r, y+h-r), (x+w-r, y+h-r)]:
        cv2.circle(overlay, (cx_, cy_), r, col, -1)
    cv2.addWeighted(overlay, alpha, canvas, 1.0 - alpha, 0, canvas)

    # Border
    border_col = C_ACCENT if highlighted else (40, 45, 60)
    cv2.rectangle(canvas, (x + 1, y + 1), (x + w - 1, y + h - 1),
                  border_col, 1)

    # Key badge
    badge_size = 34
    bx = x + 18; by = y + h // 2 - badge_size // 2
    cv2.rectangle(canvas, (bx, by), (bx + badge_size, by + badge_size),
                  C_ACCENT if highlighted else (50, 60, 50), -1)
    (kw, kh), _ = cv2.getTextSize(key_char, FONT, 0.7, 2)
    cv2.putText(canvas, key_char,
                (bx + (badge_size - kw) // 2, by + (badge_size + kh) // 2),
                FONT, 0.7, C_BG if highlighted else C_DIM, 2, cv2.LINE_AA)

    # Label
    cv2.putText(canvas, label, (bx + badge_size + 14, y + h // 2 - 6),
                FONT, 0.62, C_WHITE if highlighted else C_TEXT, 1, cv2.LINE_AA)

    # Description
    cv2.putText(canvas, desc,  (bx + badge_size + 14, y + h // 2 + 18),
                FONT, 0.42, C_DIM, 1, cv2.LINE_AA)


def show_welcome_screen(sw, sh):
    """
    Welcome / calibration-choice screen.
    Returns: 'r' reuse, 'c' calibrate, 'q' quit
    """
    _AnimState.start = time.time()
    _ensure_fullscreen()

    has_cal = calibration_exists()
    result  = None

    while result is None:
        t = _AnimState.t()
        canvas = np.full((sh, sw, 3), C_BG, dtype=np.uint8)
        _draw_bg(canvas, sw, sh)

        # Decorative top strip
        cv2.rectangle(canvas, (0, 0), (sw, 4), C_ACCENT, -1)

        _draw_logo(canvas, sw, sh)

        if has_cal:
            # Two option boxes
            box_w = min(480, sw // 2 - 40)
            box_h = 72
            total = box_w * 2 + 40
            bx1   = (sw - total) // 2
            bx2   = bx1 + box_w + 40
            by    = sh // 2 + 20

            _draw_option_box(canvas, bx1, by, box_w, box_h,
                             "R", "Reuse Calibration",
                             "Use saved session  (faster)",
                             highlighted=True, anim_t=t)
            _draw_option_box(canvas, bx2, by, box_w, box_h,
                             "C", "Recalibrate",
                             "Re-run 16-point calibration")

            note = "Recalibrate if you moved or lighting changed"
            (nw, _), _ = cv2.getTextSize(note, FONT, 0.42, 1)
            cv2.putText(canvas, note, ((sw - nw) // 2, by + box_h + 30),
                        FONT, 0.42, C_DIM, 1, cv2.LINE_AA)

        else:
            # First run — single button
            box_w = min(440, sw // 2)
            box_h = 72
            bx    = (sw - box_w) // 2
            by    = sh // 2 + 20
            _draw_option_box(canvas, bx, by, box_w, box_h,
                             "SPACE", "Start Calibration",
                             "Look at 16 dots — takes ~45 seconds",
                             highlighted=True, anim_t=t)

            tip = "Sit comfortably  ·  face the camera  ·  good lighting"
            (tw2, _), _ = cv2.getTextSize(tip, FONT, 0.42, 1)
            cv2.putText(canvas, tip, ((sw - tw2) // 2, by + box_h + 30),
                        FONT, 0.42, C_DIM, 1, cv2.LINE_AA)

        # Quit hint bottom
        q_hint = "Q  quit"
        cv2.putText(canvas, q_hint, (sw - 90, sh - 18),
                    FONT, 0.40, (50, 55, 65), 1, cv2.LINE_AA)

        # Version tag
        cv2.putText(canvas, "v2.0", (14, sh - 12),
                    FONT, 0.38, (40, 45, 55), 1, cv2.LINE_AA)

        # Decorative bottom strip
        cv2.rectangle(canvas, (0, sh - 4), (sw, sh), C_ACCENT, -1)

        cv2.imshow(WINDOW_NAME, canvas)
        key = cv2.waitKey(30) & 0xFF

        if key in (ord('q'), 27):
            result = 'q'
        elif has_cal and key == ord('r'):
            result = 'r'
        elif has_cal and key == ord('c'):
            result = 'c'
        elif not has_cal and key == ord(' '):
            result = 'c'

    return result


def show_calibration_prompt(sw, sh):
    """Brief animated screen before calibration starts. Returns True to proceed."""
    _AnimState.start = time.time()
    _ensure_fullscreen()

    start = time.time()
    while time.time() - start < 0.3:
        cv2.waitKey(1)

    while True:
        t = _AnimState.t()
        canvas = np.full((sh, sw, 3), C_BG, dtype=np.uint8)
        _draw_bg(canvas, sw, sh)
        cv2.rectangle(canvas, (0, 0), (sw, 4), C_ACCENT, -1)

        _draw_logo(canvas, sw, sh, cy_offset=-40)

        # Instructions card
        cw = min(620, sw - 80)
        ch = 220
        cx2 = (sw - cw) // 2
        cy2 = sh // 2 - 20

        overlay = canvas.copy()
        cv2.rectangle(overlay, (cx2, cy2), (cx2 + cw, cy2 + ch), C_PANEL, -1)
        cv2.addWeighted(overlay, 0.85, canvas, 0.15, 0, canvas)
        cv2.rectangle(canvas, (cx2, cy2), (cx2 + cw, cy2 + ch), (35, 40, 55), 1)
        cv2.line(canvas, (cx2, cy2), (cx2 + cw, cy2), C_ACCENT, 2)

        lines = [
            ("16 dots will appear across the screen.", 0.55, C_TEXT),
            ("Look at each dot — hold until the circle fills.", 0.55, C_TEXT),
            ("Keep your head still throughout.", 0.55, C_DIM),
            ("", 0, C_BG),
            ("Press  SPACE  to begin  ·  ESC to cancel", 0.55, C_ACCENT),
        ]
        ly = cy2 + 34
        for txt, sc, col in lines:
            if sc == 0:
                ly += 10
                continue
            (tw, th), _ = cv2.getTextSize(txt, FONT, sc, 1)
            cv2.putText(canvas, txt, ((sw - tw) // 2, ly),
                        FONT, sc, col, 1, cv2.LINE_AA)
            ly += th + 14

        cv2.imshow(WINDOW_NAME, canvas)
        key = cv2.waitKey(30) & 0xFF
        if key == ord(' '):
            return True
        if key in (ord('q'), 27):
            return False


def show_post_calibration(sw, sh, success=True):
    """Brief success/failure splash after calibration finishes."""
    _AnimState.start = time.time()
    _ensure_fullscreen()
    msg   = "Calibration Complete!" if success else "Calibration Failed"
    sub   = "Loading keyboard…"    if success else "Press any key to return"
    color = C_ACCENT if success else C_WARN
    start = time.time()

    while time.time() - start < (1.5 if success else 3.0):
        canvas = np.full((sh, sw, 3), C_BG, dtype=np.uint8)
        _draw_bg(canvas, sw, sh)
        cv2.rectangle(canvas, (0, 0), (sw, 4), color, -1)
        cv2.rectangle(canvas, (0, sh - 4), (sw, sh), color, -1)

        t = _AnimState.t()
        r = int(40 + 10 * abs(math.sin(t * 3)))
        cv2.circle(canvas, (sw // 2, sh // 2 - 70), r, color, 3, cv2.LINE_AA)
        if success:
            # Draw checkmark inside circle
            pts = np.array([
                [sw // 2 - 14, sh // 2 - 70],
                [sw // 2 - 4,  sh // 2 - 58],
                [sw // 2 + 18, sh // 2 - 90],
            ], dtype=np.int32)
            cv2.polylines(canvas, [pts], False, color, 3, cv2.LINE_AA)

        (mw, _), _ = cv2.getTextSize(msg, FONT, 1.1, 2)
        cv2.putText(canvas, msg, ((sw - mw) // 2, sh // 2 + 10),
                    FONT, 1.1, color, 2, cv2.LINE_AA)
        (sw2, _), _ = cv2.getTextSize(sub, FONT, 0.55, 1)
        cv2.putText(canvas, sub, ((sw - sw2) // 2, sh // 2 + 50),
                    FONT, 0.55, C_DIM, 1, cv2.LINE_AA)

        cv2.imshow(WINDOW_NAME, canvas)
        key = cv2.waitKey(30) & 0xFF
        if not success and key != 255:
            break


# ── Main loop ──────────────────────────────────────────────────────────────────

def main():
    sw, sh = get_screen_size()
    print(f"Screen: {sw}×{sh}")

    tracker = GazeTracker()
    tracker.open()
    model   = None

    # ── Decision screen ────────────────────────────────────────────────────────
    choice = show_welcome_screen(sw, sh)

    if choice == 'q':
        tracker.close()
        cv2.destroyAllWindows()
        return

    if choice == 'r':
        model = load_calibration()
        if model is None:
            print("WARNING: Saved calibration file missing — recalibrating.")
            choice = 'c'

    if choice == 'c':
        proceed = show_calibration_prompt(sw, sh)
        if not proceed:
            tracker.close()
            cv2.destroyAllWindows()
            return

        ok = run_calibration(tracker, sw, sh)
        show_post_calibration(sw, sh, success=ok)

        if not ok:
            tracker.close()
            cv2.destroyAllWindows()
            return

        model = load_calibration()
        if model is None:
            print("ERROR: Model missing after calibration.")
            tracker.close()
            cv2.destroyAllWindows()
            return

    # ── Keyboard session ───────────────────────────────────────────────────────
    predictor = GazePredictor(model)
    keyboard  = KeyboardUI(sw, sh)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam.")
        tracker.close()
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    _ensure_fullscreen()
    print("GazeType running.  R=recalibrate  Q/ESC=quit")

    prev_t = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)

        # ── Single MediaPipe inference per frame ───────────────────────────────
        landmarks = tracker.process_frame(frame)
        features  = tracker.get_features_from_landmarks(landmarks)

        # ── Head stability ─────────────────────────────────────────────────────
        head_stable = tracker.check_head_stability(landmarks)

        # ── Gaze prediction ────────────────────────────────────────────────────
        gaze_xy = None
        if features is not None:
            raw = predictor.predict(features)
            if raw is not None:
                gaze_xy = predictor.clamp_to_screen(raw, sw, sh)

        # ── Keyboard update + render ───────────────────────────────────────────
        keyboard.update(gaze_xy, head_stable)
        frame_out = keyboard.render(gaze_xy)

        # FPS counter (top-right, subtle)
        now = time.time()
        fps = 1.0 / max(now - prev_t, 1e-6)
        prev_t = now
        cv2.putText(frame_out, f"{fps:.0f}fps",
                    (sw - 58, sh - 12), FONT, 0.38, (35, 40, 50), 1)

        cv2.imshow(WINDOW_NAME, frame_out)

        # ── Key handling ───────────────────────────────────────────────────────
        k = cv2.waitKey(1) & 0xFF
        if k in (ord('q'), 27):
            break
        elif k == ord('r'):
            print("Recalibrating…")
            predictor.reset()
            cap.release()
            proceed = show_calibration_prompt(sw, sh)
            if proceed:
                ok = run_calibration(tracker, sw, sh)
                show_post_calibration(sw, sh, success=ok)
                if ok:
                    new_model = load_calibration()
                    if new_model:
                        predictor.model = new_model
                        predictor.reset()
                        print("Recalibration complete.")
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            _ensure_fullscreen()

    cap.release()
    tracker.close()
    cv2.destroyAllWindows()
    print("GazeType exited.")


if __name__ == "__main__":
    main()
