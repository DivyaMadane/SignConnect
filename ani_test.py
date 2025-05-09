import numpy as np
import cv2
import mediapipe as mp
from tensorflow.keras.models import load_model

# Load model and classes
model = load_model("animal_signs_model.h5")
classes = np.load("animal_label_classes.npy")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2)

cap = cv2.VideoCapture(0)

sequence = []
sequence_length = 30

while True:
    ret, frame = cap.read()
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    row = []
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                row.extend([lm.x, lm.y, lm.z])
    while len(row) < 126:
        row.append(0.0)
    
    if len(row) == 126:
        sequence.append(row)

    if len(sequence) == sequence_length:
        X_input = np.array(sequence).reshape(1, sequence_length, 126)
        prediction = model.predict(X_input)
        gesture = classes[np.argmax(prediction)]
        cv2.putText(frame, f'Animal Sign: {gesture}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        sequence = []

    cv2.imshow("Animal Sign Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
