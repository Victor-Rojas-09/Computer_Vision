import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# -------------------------------
# Load the model
# -------------------------------
MODEL_PATH = "models/hand_landmarker.task"

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2
)

detector = vision.HandLandmarker.create_from_options(options)


# Initialize webcam
cap = cv2.VideoCapture(0)

# Create window
cv2.namedWindow("Hand Detection", cv2.WINDOW_NORMAL)

# -------------------------------
# Main loop
# -------------------------------

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame horizontally (mirror effect)
    frame = cv2.flip(frame, 1)

    # Convert BGR to RGB (required by MediaPipe)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert to MediaPipe Image format
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # Run hand detection
    result = detector.detect(mp_image)

    # -------------------------------
    # Draw landmarks
    # -------------------------------

    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:
            for landmark in hand_landmarks:
                h, w, _ = frame.shape
                x = int(landmark.x * w)
                y = int(landmark.y * h)

                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    # Show frame
    cv2.imshow("Hand Detection", frame)

    # Press 'q' or 'x' to exit
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == ord('x'):
        break

    # Detect if window was closed manually (clicking X)
    if cv2.getWindowProperty("Hand Detection", cv2.WND_PROP_VISIBLE) < 1:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()