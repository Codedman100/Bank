import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import tensorflow as tf

# Initialize text-to-speech engine
tts_engine = pyttsx3.init()

# Load pre-trained MobileNet SSD model for object detection
MODEL_PATH = "ssd_mobilenet_v2_fpnlite.tflite"
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Initialize MediaPipe Hand Tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Labels for object detection
LABELS = ["background", "person", "bicycle", "car", "motorbike", "airplane", "bus", "train", "truck", "boat",
          "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse"]


# Function to perform object detection
def detect_objects(frame1):
    input_data = cv2.resize(frame1, (300, 300))
    input_data = np.expand_dims(input_data, axis=0).astype(np.uint8)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    boxes1 = interpreter.get_tensor(output_details[0]['index'])[0]
    class_ids1 = interpreter.get_tensor(output_details[1]['index'])[0]
    scores1 = interpreter.get_tensor(output_details[2]['index'])[0]

    return boxes1, class_ids1, scores1


# Function to speak the detected object
def speak(text):
    tts_engine.say(text)
    tts_engine.runAndWait()


# Start video stream
cap = cv2.VideoCapture(0)
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Unable to access camera.")
        break

    # Convert frame to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    # Perform hand tracking
    finger_x, finger_y = None, None
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            finger_x, finger_y = int(hand_landmarks.landmark[8].x * frame.shape[1]), int(
                hand_landmarks.landmark[8].y * frame.shape[0])

    # Perform object detection
    boxes, class_ids, scores = detect_objects(frame)

    for i, score in enumerate(scores):
        if score > 0.5:  # Confidence threshold
            box = boxes[i]
            x1 = int(box[1] * frame.shape[1])
            y1 = int(box[0] * frame.shape[0])
            x2 = int(box[3] * frame.shape[1])
            y2 = int(box[2] * frame.shape[0])

            # Check if the finger is pointing at the object
            if finger_x and finger_y and x1 <= finger_x <= x2 and y1 <= finger_y <= y2:
                label = LABELS[int(class_ids[i])]
                speak(f"{label}")
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Accessible Vision", frame)

    # Exit condition
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
