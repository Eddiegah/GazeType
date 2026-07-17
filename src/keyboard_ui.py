"""
keyboard_ui.py — Premium OpenCV-rendered dwell-time keyboard with rich visuals.

DESIGN:
    Dark-glass aesthetic with gradient key fills, smooth dwell arc,
    animated cursor, typed-text display, and status feedback — all
    rendered purely with OpenCV (no extra GUI library).

DWELL SELECTION:
    Gaze must stay within a key for DWELL_SECONDS to trigger a press.
    A circular arc progress indicator grows around the key center.
    After a press fires there's a brief COOLDOWN to prevent double-fires.
"""

import time
import math
import cv2
import numpy as np

# ── Timing ─────────────────────────────────────────────────────────────────────
DWELL_SECONDS    = 1.0
COOLDOWN_SECONDS = 0.5

# ── Layout rows ───────────────────────────────────────────────────────────────
ROWS = [
    list("QWERTYUIOP"),
    list("ASDFGHJKL"),
    list("ZXCVBNM"),
    ["BKSP", "SPACE", "CLEAR"],
]

# ── Palette — dark glass theme ────────────────────────────────────────────────
# All colors in BGR
C_BG            = (10,  10,  16)    # near-black with blue tint
C_PANEL         = (18,  18,  28)    # slightly lighter for panels
C_KEY_BASE      = (32,  34,  44)    # key resting state
C_KEY_HOVER     = (44,  56,  44)    # key with gaze nearby
C_KEY_EDGE_TOP  = (70,  72,  90)    # highlight edge (top-left of key)
C_KEY_EDGE_BOT  = (15,  15,  22)    # shadow edge (bottom-right of key)
C_DWELL_FILL    = (24,  80,  40)    # dwell progress fill inside key
C_DWELL_ARC     = (0,  200, 120)    # arc ring color
C_PRESSED       = (30, 180,  90)    # flashed on press
C_LABEL         = (210, 215, 225)   # key letter color
C_LABEL_SPECIAL = (140, 190, 140)   # BKSP / SPACE / CLEAR text
C_OUTPUT_BG     = (8,    8,  14)    # output bar background
C_OUTPUT_TEXT   = (0,  230, 140)    # typed text color
C_PLACEHOLDER   = (50,  55,  65)    # placeholder text
C_CURSOR_RING   = (0,  160, 255)    # gaze cursor outer ring
C_CURSOR_CORE   = (0,  210, 255)    # gaze cursor center
C_CURSOR_GLOW   = (0,   80, 120)    # gaze cursor soft glow
C_ACCENT        = (0,  200, 120)    # UI accent color (matches dwell arc)
C_WARNING_BG    = (14,  10,  30)
C_WARNING_TEXT  = (60, 100, 220)
C_STATUS_OK     = (0,  170, 100)
C_DIVIDER       = (28,  30,  42)
C_TITLE         = (0,  200, 120)

# ── Proportions ────────────────────────────────────────────────────────────────
OUTPUT_H_FRAC  = 0.13   # output text area
TOPBAR_H_FRAC  = 0.06   # title / status bar
KB_H_FRAC      = 0.60   # keyboard rows
KEY_GAP        = 5      # pixels between keys
KEY_RADIUS     = 6      # rounded corner radius

# ── Font shorthand ─────────────────────────────────────────────────────────────
FONT = cv2.FONT_HERSHEY_SIMPLEX


class Key:
    __slots__ = ("label", "x", "y", "w", "h")

    def __init__(self, label, x, y, w, h):
        self.label = label
        self.x = x; self.y = y; self.w = w; self.h = h

    def contains(self, px, py):
        return self.x <= px < self.x + self.w and self.y <= py < self.y + self.h

    @property
    def cx(self): return self.x + self.w // 2

    @property
    def cy(self): return self.y + self.h // 2


class KeyboardUI:
    """
    Full keyboard UI manager.  Call update() each frame, then render().
    """

    def __init__(self, screen_w: int, screen_h: int):
        self.sw = screen_w
        self.sh = screen_h
        self.typed_text   = ""
        self.status_msg   = ""
        self.status_ok    = True

        self._dwell_key    = None
        self._dwell_start  = None
        self._cooldown_end = 0.0
        self._last_label   = None
        self._last_press_t = 0.0
        self._blink_phase  = 0.0  # cursor animation

        self.keys = self._build_layout()

    # ── Layout ─────────────────────────────────────────────────────────────────

    def _build_layout(self):
        sw, sh = self.sw, self.sh
        oh  = int(sh * OUTPUT_H_FRAC)
        tbh = int(sh * TOPBAR_H_FRAC)
        kbh = int(sh * KB_H_FRAC)
        kb_top = sh - kbh
        pad = KEY_GAP

        # Thin decorative accent line above keyboard
        self._kb_top = kb_top

        row_h = (kbh - pad * (len(ROWS) + 1)) // len(ROWS)
        keys  = []

        for ri, row in enumerate(ROWS):
            ry = kb_top + pad + ri * (row_h + pad)

            if ri < 3:
                n      = len(row)
                indent = ri * int(sw * 0.022)
                total  = sw - 2 * pad - 2 * indent
                kw     = (total - (n - 1) * pad) // n
                for ci, label in enumerate(row):
                    kx = pad + indent + ci * (kw + pad)
                    keys.append(Key(label, kx, ry, kw, row_h))
            else:
                # Special row
                bw = int(sw * 0.14)
                cw = int(sw * 0.13)
                sp = sw - 2 * pad - bw - cw - 2 * pad * 2
                specs = [("BKSP", bw), ("SPACE", sp), ("CLEAR", cw)]
                kx = pad
                for label, w in specs:
                    keys.append(Key(label, kx, ry, w, row_h))
                    kx += w + pad * 2

        return keys

    # ── Per-frame update ────────────────────────────────────────────────────────

    def update(self, gaze_xy, head_stable=True):
        """
        Process one frame. Returns pressed key label or None.
        """
        now = time.time()
        self._blink_phase = (now * 2.5) % (2 * math.pi)

        if gaze_xy is None:
            self.status_msg = "Face not detected — look toward the camera"
            self.status_ok  = False
            self._dwell_key   = None
            self._dwell_start = None
            return None

        if head_stable is False:
            self.status_msg = "Head moved — recalibrate if cursor drifts  (R)"
            self.status_ok  = False
        else:
            self.status_msg = ""
            self.status_ok  = True

        gx, gy = int(gaze_xy[0]), int(gaze_xy[1])

        hovered = None
        for key in self.keys:
            if key.contains(gx, gy):
                hovered = key
                break

        if hovered is None:
            self._dwell_key   = None
            self._dwell_start = None
        elif hovered is not self._dwell_key:
            self._dwell_key   = hovered
            self._dwell_start = now
        else:
            elapsed = now - self._dwell_start
            if elapsed >= DWELL_SECONDS and now >= self._cooldown_end:
                label = hovered.label
                self._last_label   = label
                self._last_press_t = now
                self._cooldown_end = now + COOLDOWN_SECONDS
                self._dwell_key    = None
                self._dwell_start  = None
                self._apply(label)
                return label

        return None

    def _apply(self, label):
        if label == "BKSP":
            self.typed_text = self.typed_text[:-1]
        elif label == "SPACE":
            self.typed_text += " "
        elif label == "CLEAR":
            self.typed_text = ""
        else:
            self.typed_text += label

    # ── Render ─────────────────────────────────────────────────────────────────

    def render(self, gaze_xy=None):
        """Render full UI to a (sh × sw × 3) BGR image."""
        canvas = np.full((self.sh, self.sw, 3), C_BG, dtype=np.uint8)
        self._draw_output_bar(canvas)
        self._draw_topbar(canvas)
        self._draw_keyboard(canvas)
        if gaze_xy is not None:
            self._draw_cursor(canvas, gaze_xy)
        return canvas

    # ── Output bar ─────────────────────────────────────────────────────────────

    def _draw_output_bar(self, canvas):
        oh = int(self.sh * OUTPUT_H_FRAC)
        sw = self.sw

        # Background with subtle gradient (draw two rects)
        cv2.rectangle(canvas, (0, 0), (sw, oh), C_OUTPUT_BG, -1)

        # Bottom border — accent line
        cv2.line(canvas, (0, oh - 1), (sw, oh - 1), C_ACCENT, 2)

        # Label
        cv2.putText(canvas, "OUTPUT", (12, 18),
                    FONT, 0.38, (50, 60, 50), 1, cv2.LINE_AA)

        # Cursor blink after last char
        blink_on = math.sin(self._blink_phase) > 0

        display = self.typed_text if self.typed_text else ""
        show    = display + ("|" if blink_on else " ")
        color   = C_OUTPUT_TEXT if self.typed_text else C_PLACEHOLDER

        if not self.typed_text:
            show  = "start looking at keys to type…"
            color = C_PLACEHOLDER

        # Truncate left to fit
        scale, thick = 1.1, 2
        max_w = sw - 80
        while True:
            (tw, th), _ = cv2.getTextSize(show, FONT, scale, thick)
            if tw <= max_w or len(show) < 2:
                break
            show = show[1:]

        ty = oh // 2 + th // 2
        cv2.putText(canvas, show, (16, ty), FONT, scale, color, thick, cv2.LINE_AA)

    # ── Top bar ─────────────────────────────────────────────────────────────────

    def _draw_topbar(self, canvas):
        oh  = int(self.sh * OUTPUT_H_FRAC)
        tbh = int(self.sh * TOPBAR_H_FRAC)
        y1  = oh
        y2  = oh + tbh
        sw  = self.sw

        cv2.rectangle(canvas, (0, y1), (sw, y2), C_PANEL, -1)
        cv2.line(canvas, (0, y2), (sw, y2), C_DIVIDER, 1)

        # Logo / title left side
        cv2.putText(canvas, "GAZETYPE", (14, y1 + tbh - 10),
                    FONT, 0.55, C_TITLE, 1, cv2.LINE_AA)

        # Hint text right side
        hint = "R = recalibrate    Q / ESC = quit"
        (hw, _), _ = cv2.getTextSize(hint, FONT, 0.40, 1)
        cv2.putText(canvas, hint, (sw - hw - 14, y1 + tbh - 10),
                    FONT, 0.40, (70, 75, 80), 1, cv2.LINE_AA)

        # Status / warning centered
        if self.status_msg:
            sc = C_STATUS_OK if self.status_ok else C_WARNING_TEXT
            (mw, _), _ = cv2.getTextSize(self.status_msg, FONT, 0.42, 1)
            mx = (sw - mw) // 2
            cv2.putText(canvas, self.status_msg, (mx, y1 + tbh - 10),
                        FONT, 0.42, sc, 1, cv2.LINE_AA)

    # ── Keyboard ────────────────────────────────────────────────────────────────

    def _draw_keyboard(self, canvas):
        now = time.time()

        # Accent line above keyboard
        kbt = self._kb_top
        cv2.line(canvas, (0, kbt), (self.sw, kbt), C_ACCENT, 1)

        for key in self.keys:
            self._draw_key(canvas, key, now)

    def _draw_key(self, canvas, key: Key, now: float):
        x, y, w, h = key.x, key.y, key.w, key.h
        cx, cy = key.cx, key.cy
        is_special = len(key.label) > 1

        # ── Dwell state for this key ──────────────────────────────────────────
        is_dwell   = (self._dwell_key is key)
        dwell_prog = 0.0
        if is_dwell and self._dwell_start is not None:
            dwell_prog = min((now - self._dwell_start) / DWELL_SECONDS, 1.0)

        just_pressed = (key.label == self._last_label and
                        now - self._last_press_t < COOLDOWN_SECONDS)

        # ── Background fill ───────────────────────────────────────────────────
        if just_pressed:
            fill = C_PRESSED
        elif is_dwell:
            # Blend base → dwell fill based on progress
            fill = _blend(C_KEY_HOVER, C_DWELL_FILL, dwell_prog * 0.6)
        else:
            fill = C_KEY_HOVER if is_dwell else C_KEY_BASE

        _draw_rounded_rect(canvas, x, y, w, h, KEY_RADIUS, fill)

        # ── Dwell progress fill (left → right) ───────────────────────────────
        if dwell_prog > 0 and not just_pressed:
            fw = int(w * dwell_prog)
            if fw > KEY_RADIUS * 2:
                _draw_rounded_rect(canvas, x, y, fw, h, KEY_RADIUS,
                                   C_DWELL_FILL, alpha=0.55)

        # ── Circular arc indicator centred on key ─────────────────────────────
        if is_dwell and dwell_prog > 0:
            arc_r = min(w, h) // 2 - 4
            if arc_r > 6:
                angle = int(360 * dwell_prog)
                cv2.ellipse(canvas, (cx, cy), (arc_r, arc_r),
                            -90, 0, angle, C_DWELL_ARC, 2, cv2.LINE_AA)

        # ── Top-left highlight / bottom-right shadow (bevel look) ────────────
        if not just_pressed:
            _draw_rounded_rect_outline(canvas, x, y, w, h, KEY_RADIUS,
                                       C_KEY_EDGE_TOP, C_KEY_EDGE_BOT)
        else:
            _draw_rounded_rect_outline(canvas, x, y, w, h, KEY_RADIUS,
                                       C_ACCENT, C_ACCENT)

        # ── Label ─────────────────────────────────────────────────────────────
        label = key.label
        if is_special:
            scale, thick = 0.50, 1
            lcolor = C_LABEL_SPECIAL
        else:
            scale, thick = 0.65, 2
            lcolor = C_LABEL

        if just_pressed:
            lcolor = (255, 255, 255)

        (tw, th), _ = cv2.getTextSize(label, FONT, scale, thick)
        tx = x + (w - tw) // 2
        ty = y + (h + th) // 2
        cv2.putText(canvas, label, (tx, ty), FONT, scale, lcolor, thick,
                    cv2.LINE_AA)

    # ── Cursor ──────────────────────────────────────────────────────────────────

    def _draw_cursor(self, canvas, gaze_xy):
        gx, gy = int(gaze_xy[0]), int(gaze_xy[1])

        # Soft glow (large semi-transparent circle)
        _draw_circle_alpha(canvas, gx, gy, 22, C_CURSOR_GLOW, 0.25)

        # Outer ring — pulses slightly
        pulse_r = 13 + int(2 * math.sin(self._blink_phase))
        cv2.circle(canvas, (gx, gy), pulse_r, C_CURSOR_RING, 2, cv2.LINE_AA)

        # Inner filled dot
        cv2.circle(canvas, (gx, gy), 5, C_CURSOR_CORE, -1, cv2.LINE_AA)

        # Cross-hair lines for precision feedback
        half = 9
        cv2.line(canvas, (gx - half, gy), (gx - 7, gy), C_CURSOR_RING, 1)
        cv2.line(canvas, (gx + 7, gy),    (gx + half, gy), C_CURSOR_RING, 1)
        cv2.line(canvas, (gx, gy - half), (gx, gy - 7), C_CURSOR_RING, 1)
        cv2.line(canvas, (gx, gy + 7),    (gx, gy + half), C_CURSOR_RING, 1)


# ── Drawing utilities ──────────────────────────────────────────────────────────

def _draw_rounded_rect(canvas, x, y, w, h, r, color, alpha=1.0):
    """Fill a rounded rectangle. alpha < 1 does a simple blend overlay."""
    if alpha >= 1.0:
        # Four rects + four circles
        cv2.rectangle(canvas, (x + r, y),     (x + w - r, y + h), color, -1)
        cv2.rectangle(canvas, (x,     y + r), (x + w,     y + h - r), color, -1)
        cv2.circle(canvas, (x + r,     y + r),     r, color, -1)
        cv2.circle(canvas, (x + w - r, y + r),     r, color, -1)
        cv2.circle(canvas, (x + r,     y + h - r), r, color, -1)
        cv2.circle(canvas, (x + w - r, y + h - r), r, color, -1)
    else:
        overlay = canvas.copy()
        _draw_rounded_rect(overlay, x, y, w, h, r, color, alpha=1.0)
        cv2.addWeighted(overlay, alpha, canvas, 1.0 - alpha, 0, canvas)


def _draw_rounded_rect_outline(canvas, x, y, w, h, r, top_col, bot_col):
    """Draw a two-tone bevel outline: top/left in top_col, bottom/right in bot_col."""
    # Top and left edges
    cv2.line(canvas, (x + r, y),     (x + w - r, y),     top_col, 1)
    cv2.line(canvas, (x,     y + r), (x,         y + h - r), top_col, 1)
    # Bottom and right edges
    cv2.line(canvas, (x + r, y + h), (x + w - r, y + h), bot_col, 1)
    cv2.line(canvas, (x + w, y + r), (x + w,     y + h - r), bot_col, 1)


def _draw_circle_alpha(canvas, cx, cy, r, color, alpha):
    overlay = canvas.copy()
    cv2.circle(overlay, (cx, cy), r, color, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, canvas, 1.0 - alpha, 0, canvas)


def _blend(c1, c2, t):
    """Linear interpolate between two BGR tuples by factor t ∈ [0,1]."""
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))
