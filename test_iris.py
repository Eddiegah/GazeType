"""
test_iris.py — Standalone iris landmark test.

Run this FIRST to confirm MediaPipe can see and track your eyes
before building the full application.

Usage:
    venv\Scripts\python.exe test_iris.py

Controls:
    Q  — quit

What you should see:
    - Green dots on your face mesh
    - Bright CYAN dots on your iris landmarks (both eyes)
    - The window title showing FPS

If the iris dots are missing but face dots appear, make sure your
lighting is adequate and you're facing the camera directly.
"""

import cv2
import mediapipe as mp
import time

# MediaPipe setup
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Iris landmark indices in the MediaPipe 478-point refined face mesh.
# LEFT_IRIS  = landmarks 468–471 (center + 4 edge points)
# RIGHT_IRIS = landmarks 473–476
LEFT_IRIS_INDICES  = [468, 469, 470, 471, 472]
RIGHT_IRIS_INDICES = [473, 474, 475, 476, 477]


def draw_iris_landmarks(frame, face_landmarks, h, w):
    """Draw all iris landmarks as cyan circles on the frame."""
    for idx in LEFT_IRIS_INDICES + RIGHT_IRIS_INDICES:
        lm = face_landmarks.landmark[idx]
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (cx, cy), 3, (0, 255, 255), -1)  # cyan filled dot


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam. Check that no other app is using it.")
        return

    # Set a reasonable resolution — 640×480 is fast and sufficient
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Iris landmark test running. Press Q to quit.")
    print("You should see cyan dots on your irises.")

    prev_time = time.time()

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,   # REQUIRED — enables the 10 extra iris landmarks
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:

        while True:
            ret, frame = cap.read()
            if not ret:
                print("ERROR: Failed to read frame from webcam.")
                break

            # Flip horizontally so it acts like a mirror (more intuitive)
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            # MediaPipe expects RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = face_mesh.process(rgb)
            rgb.flags.writeable = True

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Draw the full face mesh in a subtle style
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles
                            .get_default_face_mesh_tesselation_style(),
                    )
                    # Draw iris landmarks on top, larger and brighter
                    draw_iris_landmarks(frame, face_landmarks, h, w)

                status = "Face detected — iris landmarks active"
                color = (0, 200, 0)
            else:
                status = "No face detected — move closer or improve lighting"
                color = (0, 0, 200)

            # FPS counter
            now = time.time()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            cv2.putText(frame, status, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.putText(frame, "Press Q to quit", (10, h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

            cv2.imshow("GazeType — Iris Landmark Test", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("Test complete.")


if __name__ == "__main__":
    main()
