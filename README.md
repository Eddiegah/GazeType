<div align="center">

# 👁️ GazeType

### Type with your eyes. No hardware. Just a webcam.

[![Python](https://img.shields.io/badge/Python-3.9–3.12-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.21-0097A7?style=flat-square)](https://mediapipe.dev)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.10-5C3EE8?style=flat-square&logo=opencv&logoColor=white)](https://opencv.org)
[![Platform](https://img.shields.io/badge/Platform-Windows-0078D6?style=flat-square&logo=windows&logoColor=white)](https://www.microsoft.com/windows)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

**GazeType** turns your regular webcam into a hands-free typing interface.
Look at a key on the on-screen keyboard. Hold your gaze for one second.
The key fires.

No eye-tracking hardware. No mouse. No keyboard. Just your eyes.

---

![GazeType Demo](https://raw.githubusercontent.com/Eddiegah/GazeType/main/assets/demo.gif)

*▲ The gaze cursor follows your eyes in real time. Dwell on a key to type it.*

</div>

---

## ✨ What makes it work

GazeType uses **MediaPipe's iris landmark model** (478 facial landmarks, 10 of which are iris-specific) to track exactly where your eyes are pointing inside your eye socket. A **16-point calibration routine** maps those eye positions to your screen coordinates using a machine-learning regression model. The result: a real-time cursor driven entirely by your gaze.

Every accuracy detail was engineered carefully:

| Improvement | What it does for you |
|---|---|
| **10-feature vector** (was 4) | Captures gaze angle from multiple reference frames — far more robust to head tilt |
| **16-point calibration** (was 9) | Better coverage of screen edges and corners where error tends to be highest |
| **Keeps all raw frames** (~400 samples, was 9) | The regression model has real data to work with instead of 9 averaged points |
| **Gradient Boosting model** (was MLP) | Ensemble trees generalize better with moderate sample counts, no tuning needed |
| **Adaptive EMA smoother** (was flat average) | Snappy on large gaze shifts, smooth on steady fixations — best of both worlds |
| **Outlier rejection** | Blink artifacts dropped per calibration point before model fitting |
| **Scale normalization (IPD)** | Works at different sitting distances — divides by inter-pupil distance |
| **Single MediaPipe inference/frame** | No wasted computation — landmarks processed once and shared |

---

## 🚀 One-Command Setup

> **Requires Python 3.9–3.12.** Python 3.13+ is not supported by MediaPipe.

```bash
# 1. Clone
git clone https://github.com/Eddiegah/GazeType.git
cd GazeType

# 2. Create virtual environment with Python 3.11
py -3.11 -m venv venv
venv\Scripts\activate

# 3. Install everything
pip install -r requirements.txt

# 4. Verify (should print "All imports OK")
python -c "import mediapipe; import cv2; import numpy; import sklearn; print('All imports OK')"
```

That's it. No extra downloads, no special hardware, no configuration files.

---

## 🎯 Test Your Camera First

Before launching the full app, confirm MediaPipe can see your eyes:

```bash
venv\Scripts\python.exe test_iris.py
```

You should see **cyan dots tracking your irises** as you move your eyes. If that works, you're ready.

---

## ▶️ Run GazeType

```bash
venv\Scripts\python.exe src\main.py
```

### First launch
1. Press **C** to start the 16-point calibration
2. Look at each glowing dot as it appears — hold for ~2.5 seconds
3. The on-screen keyboard appears automatically

### After that
Press **R** on launch to reuse your saved calibration. Calibration is saved to disk so you never start from scratch unless you want to.

---

## ⌨️ How to Type

| Action | What to do |
|---|---|
| **Select a key** | Look at it for ~1 second (watch the arc fill up) |
| **Space** | Dwell on the `SPACE` bar |
| **Backspace** | Dwell on `BKSP` |
| **Clear all** | Dwell on `CLEAR` |
| **Recalibrate** | Press `R` on keyboard anytime |
| **Quit** | Press `Q` or `ESC` |

---

## 🎮 Runtime Controls

| Key | Action |
|---|---|
| `R` | Recalibrate (no restart needed) |
| `Q` / `ESC` | Quit |

---

## 📁 Project Structure

```
GazeType/
├── src/
│   ├── main.py            → App entry point, animated UI screens, main loop
│   ├── gaze_tracker.py    → MediaPipe iris extraction, 10-feature vector, head stability
│   ├── calibration.py     → 16-point calibration, gradient boosting model fitting
│   ├── gaze_predictor.py  → Adaptive EMA smoother, outlier gate, real-time prediction
│   └── keyboard_ui.py     → Dark-glass QWERTY keyboard, dwell arcs, animated cursor
├── calibration_data/      → Saved calibration model (gitignored — yours, not committed)
├── test_iris.py           → Standalone iris landmark verification
├── requirements.txt
└── README.md
```

---

## 🔧 Troubleshooting

<details>
<summary><strong>DLL load failed when importing mediapipe or cv2</strong></summary>

Install the Microsoft Visual C++ 2015–2022 Redistributable:
- [x64 (64-bit)](https://aka.ms/vs/17/release/vc_redist.x64.exe)
- [x86 (32-bit)](https://aka.ms/vs/17/release/vc_redist.x86.exe)

Restart your computer and retry.
</details>

<details>
<summary><strong>Webcam won't open</strong></summary>

Close any other app using your camera (Teams, Zoom, browser). If you have multiple cameras, change `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)` in `main.py` and `test_iris.py`.
</details>

<details>
<summary><strong>Cursor is consistently off in one direction</strong></summary>

Press **R** to recalibrate. The most common cause is a changed head position or different lighting since your last calibration session.
</details>

<details>
<summary><strong>Iris dots missing in test_iris.py</strong></summary>

Ensure your face is well-lit from the front. Avoid sitting with a bright window behind you. Move closer to the camera — you should fill at least 1/4 of the frame.
</details>

<details>
<summary><strong>Only works in src\ folder?</strong></summary>

Always run from the project root:
```bash
cd C:\Projects\GazeType
venv\Scripts\python.exe src\main.py
```
</details>

---

## 📏 Accuracy & Expectations

Webcam-based gaze tracking is genuinely impressive — and genuinely limited compared to dedicated hardware.

**What to expect realistically:**
- ~1–3 key width accuracy on a standard monitor at typical sitting distance
- Best performance: good frontal lighting, no glasses glare, head ~50–70 cm from monitor
- Glasses work fine in most cases; thick lenses with strong distortion may reduce accuracy
- Recalibrate after repositioning — the model is fit for your exact session

**This is not** a Tobii. It's a full gaze-based typing interface running on a $0 webcam. Within those bounds, it works.

---

## 🔬 How the Science Works

```
Webcam frame
    │
    ▼
MediaPipe Face Mesh (478 landmarks, refine_landmarks=True)
    │
    ├─ Iris center positions (landmarks 468, 473)
    ├─ Eye corner anchors (landmarks 33, 133, 263, 362...)
    │
    ▼
10-element normalized feature vector
    [iris_x/eye_width, iris_y/eye_height,       ← inner-corner reference
     same for right eye,                          ← right eye
     iris_x/eye_width from center,               ← center reference
     same for right,
     iris_y/eye_height from center, same,
     left_iris_x / inter-pupil-distance,         ← scale normalization
     right_iris_x / inter-pupil-distance]
    │
    ▼
Gradient Boosting Regressor (sklearn)
  trained on ~400 labeled samples (gaze features → screen coords)
    │
    ▼
Raw (x, y) screen coordinate prediction
    │
    ▼
Adaptive EMA smoother
  (alpha=0.30 stable, alpha=0.75 on large shift)
    │
    ▼
Clamped gaze point → keyboard dwell detection
```

---

<div align="center">

**Made with 👁️ and Python**

*Star this repo if you find it interesting — it helps others discover it.*

</div>
