import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# -------------------------------
# Load the model
# -------------------------------
MODEL_PATH = "models/face_landmarker.task"

latest_result = None

# Callback
def result_callback(result, output_image, timestamp_ms):
    global latest_result
    latest_result = result

# Init settings
BaseOptions = python.BaseOptions
FaceLandmarker = vision.FaceLandmarker
FaceLandmarkerOptions = vision.FaceLandmarkerOptions
VisionRunningMode = vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=result_callback,
    num_faces=1,
    output_face_blendshapes=True,  # Expression
    output_facial_transformation_matrixes=True  # Pose 3D
)

# Initialize webcam
cap = cv2.VideoCapture(0)

with FaceLandmarker.create_from_options(options) as landmarker:

    timestamp = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Async
        landmarker.detect_async(mp_image, timestamp)

        # Draw landmarks
        if latest_result and latest_result.face_landmarks:
            for face_landmarks in latest_result.face_landmarks:
                for lm in face_landmarks:
                    h, w, _ = frame.shape
                    x, y = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

        # Show
        cv2.imshow("Face Landmarker", frame)

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