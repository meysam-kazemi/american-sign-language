import os
import sys
import pickle
import cv2 as cv
import math
import torch
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.utils import load_config
from src.realtime_hand_landmarks import landmarkDetection

config = load_config()

TRAIN_DIR = config.get('DATA', 'train_dir')
LANDMARKS_DIR = config.get("DATA", "landmarks_dir")

def save_landmarks(all_landmarks, landmarks_dir=LANDMARKS_DIR):
    with open(landmarks_dir, 'wb') as f:
        pickle.dump(all_landmarks, f)

landmark_detection = landmarkDetection(config)
all_landmarks = []
sign_names = os.listdir(TRAIN_DIR)
os.makedirs("dataset/with_landmarks/", exist_ok=True)
os.makedirs(LANDMARKS_DIR, exist_ok=True)
for sign_name in sign_names:
    if sign_name=="nothing":
        continue
    os.makedirs("dataset/with_landmarks/"+sign_name, exist_ok=True)
    image_paths = os.listdir(os.path.join(TRAIN_DIR, sign_name))
    for i, image_path in enumerate(image_paths):
        img = cv.imread(os.path.join(TRAIN_DIR, sign_name, image_path), 1)
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        landmarks = landmark_detection.detect(img_rgb)
        landmark_detection.draw_with_landmark(img, landmarks.multi_hand_landmarks)
        all_landmarks.append(landmarks.multi_hand_landmarks)
        cv.imwrite(f"dataset/with_landmarks/{sign_name}/{str(i)}.png", img)
        loading = (50*i/len(image_paths)) + 1
        print(f'{sign_name:<5}:[{"="*math.ceil(loading):<50}] {loading:.2f}%', end='\r')
    print(f'Saving {sign_name} in {LANDMARKS_DIR+sign_name+".pkl"} ...', end='\r')
    print(f'Saved {sign_name} in {LANDMARKS_DIR+sign_name+".pkl"} ✔\n')
    save_landmarks(all_landmarks, landmarks_dir=LANDMARKS_DIR+sign_name+".pkl")


