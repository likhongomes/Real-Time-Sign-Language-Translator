import cv2
import numpy as np
import mediapipe as mp
from collections import deque
from tensorflow.keras.models import load_model

# ==========================
# CONFIG - CHANGE THESE
# ==========================

MODEL_PATH = "asl_model.h5"  # path to your trained model
# List of labels in the exact order your model was trained on
CLASS_NAMES = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
    # ...
    # Add all your ASL labels here
]

CONFIDENCE_THRESHOLD = 0.7      # min softmax score to accept prediction
SMOOTHING_WINDOW = 5            # frames to smooth prediction over

# ==========================
# LOAD MODEL
# ==========================

model = load_model(MODEL_PATH)

# How many features the model expects:
# For example, if you're using 21 hand landmarks, each with (x, y, z):
# input_dim = 21 * 3 = 63
EXPECTED_INPUT_DIM = model.input_shape[-1]
print("Model expects input dim:", EXPECTED_INPUT_DIM)

# ==========================
# MEDIAPIPE SETUP
# ==========================

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# ==========================
# HELPER: EXTRACT FEATURES
# ==========================

def extract_hand_features(results, image_width, image_height):
    """
    From MediaPipe results, build a flat vector of [x, y, z] for each landmark.
    Normalized by wrist position (landmark 0) for translation invariance.

    Returns:
        np.array of shape (FEATURE_DIM,) or None if no hand.
    """
    if not results.multi_hand_landmarks:
        return None

    # Use the first detected hand
    hand_landmarks = results.multi_hand_landmarks[0]

    # Collect landmarks
    coords = []
    wrist = hand_landmarks.landmark[0]
    wrist_x, wrist_y, wrist_z = wrist.x, wrist.y, wrist.z

    for lm in hand_landmarks.landmark:
        # Normalized coordinates (relative to wrist)
        x = lm.x - wrist_x
        y = lm.y - wrist_y
        z = lm.z - wrist_z
        coords.extend([x, y, z])

    features = np.array(coords, dtype=np.float32)

    # If the model expects a specific dimension, pad or trim
    if features.shape[0] < EXPECTED_INPUT_DIM:
        # pad with zeros
        padded = np.zeros(EXPECTED_INPUT_DIM, dtype=np.float32)
        padded[:features.shape[0]] = features
        features = padded
    elif features.shape[0] > EXPECTED_INPUT_DIM:
        # trim extra
        features = features[:EXPECTED_INPUT_DIM]

    return features

# ==========================
# MAIN LOOP
# ==========================

cap = cv2.VideoCapture(0)  # 0 = default webcam

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit(1)

prediction_history = deque(maxlen=SMOOTHING_WINDOW)
last_printed_label = None

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Flip for a selfie-view
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Convert BGR -> RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process with Mediapipe
    results = hands.process(rgb)

    # Draw hand landmarks (for visualization)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

    # Extract features
    features = extract_hand_features(results, w, h)

    predicted_label = ""
    predicted_conf = 0.0

    if features is not None:
        # Model expects shape (batch_size, input_dim)
        x = features.reshape(1, -1)

        # Predict
        probs = model.predict(x, verbose=0)[0]  # shape (num_classes,)
        predicted_index = np.argmax(probs)
        predicted_conf = float(probs[predicted_index])

        if predicted_conf >= CONFIDENCE_THRESHOLD:
            predicted_label = CLASS_NAMES[predicted_index]
            prediction_history.append(predicted_label)
        else:
            prediction_history.append("")

    # Smoothing: use most common label in recent frames
    if len(prediction_history) > 0:
        non_empty = [p for p in prediction_history if p != ""]
        if non_empty:
            # Most frequent label in history
            predicted_label_smoothed = max(set(non_empty), key=non_empty.count)
        else:
            predicted_label_smoothed = ""
    else:
        predicted_label_smoothed = ""

    # Print only when label changes and is non-empty
    if predicted_label_smoothed and predicted_label_smoothed != last_printed_label:
        print(f"Predicted ASL: {predicted_label_smoothed} (conf ~ {predicted_conf:.2f})")
        last_printed_label = predicted_label_smoothed

    # Overlay prediction on the frame
    if predicted_label_smoothed:
        cv2.putText(
            frame,
            f"{predicted_label_smoothed} ({predicted_conf:.2f})",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            3,
            cv2.LINE_AA,
        )
    else:
        cv2.putText(
            frame,
            "No confident sign",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    # Show the frame
    cv2.imshow("ASL Inference (MediaPipe + Webcam)", frame)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
hands.close()
