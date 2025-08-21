import os
import sys
import pickle
import cv2 as cv
import torch
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.utils import load_config
from src.realtime_hand_landmarks import landmarkDetection

config = load_config()

TRAIN_DIR = config.get('DATA', 'train_dir')
LANDMARKS_DIR = config.get("DATA", "landmarks_dir")

landmark_detection = landmarkDetection(config)
all_landmarks = []
sign_names = os.listdir(TRAIN_DIR)
os.makedirs("with_landmarks/", exist_ok=True)
for sign_name in sign_names:
    for i, image in enumerate(os.listdir(os.path.join(sign_names, sign_name))):
        img = cv.imread(image, 1)
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        landmarks = landmark_detection.detect(img_rgb)
        landmard_detection.draw_with_landmark(img, landmarks.multi_hand_landmarks)
        all_landmarks.append(landmarks)
        cv.imwrite("dataset/with_landmarks/"+str(i)+".csv", img)


with open(LANDMARKS_DIR, 'wb') as f:
    pickle.dump(all_landmarks, f)
