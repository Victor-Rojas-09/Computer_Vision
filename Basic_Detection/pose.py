import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# -------------------------------
# Load the model
# -------------------------------
MODEL_PATH = "models/pose_landmarker.task"

latest_result = None

# Callback
def result_callback(result, output_image, timestamp_ms):
    global latest_result
    latest_result = result

# Init settings
BaseOptions = python.BaseOptions
PoseLandmarker = vision.PoseLandmarker
PoseLandmarkerOptions = vision.PoseLandmarkerOptions
VisionRunningMode = vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=result_callback,
    num_poses=1
)

# Initialize webcam
cap = cv2.VideoCapture(0)

# -------------------------------
# Main loop
# -------------------------------
with PoseLandmarker.create_from_options(options) as landmarker:

    timestamp = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Create a MediaPipe image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Async
        landmarker.detect_async(mp_image, timestamp)

        # -------------------------------
        # Draw landmarks
        # -------------------------------

        if latest_result and latest_result.pose_landmarks:
            for pose_landmarks in latest_result.pose_landmarks:
                for lm in pose_landmarks:
                    h, w, _ = frame.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

        # show
        cv2.imshow("LIVE Pose Detection", frame)

        # Press 'q' or 'x' to exit
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('x'):
            break

        # Detect if window was closed manually (clicking X)
        if cv2.getWindowProperty("Hand Detection", cv2.WND_PROP_VISIBLE) < 1:
            break

        timestamp += 33  # ~30 FPS

# Release resources
cap.release()
cv2.destroyAllWindows()