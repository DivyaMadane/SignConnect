import cv2
import os
import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
sequence_length = 30

data = []
labels = []
root_dir = r"D:\ISL\Data_Training\animal_model\animals_dataset"  

for label in os.listdir(root_dir):
    label_path = os.path.join(root_dir, label)
    if not os.path.isdir(label_path):
        continue
    for video_file in os.listdir(label_path):
        video_path = os.path.join(label_path, video_file)
        cap = cv2.VideoCapture(video_path)
        
        sequence = []
        
        while len(sequence) < sequence_length:
            ret, frame = cap.read()
            if not ret:
                break
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(img_rgb)
            
            row = []
            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    for lm in hand_landmarks.landmark:
                        row.extend([lm.x, lm.y, lm.z])
            while len(row) < 126:
                row.append(0.0)
            
            sequence.append(row)

        if len(sequence) == sequence_length:
            data.append(sequence)
            labels.append(label)
        
        cap.release()

# Save
np.save("X_animals.npy", np.array(data))
np.save("y_animals.npy", np.array(labels))

print("✅ Preprocessing Done! Saved X_animals.npy and y_animals.npy.")
