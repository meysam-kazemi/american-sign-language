import os
import sys
import torch
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.utils import load_config
from src.realtime_hand_landmarks import landmarkDetection

config = load_config()

train_dir = config.get('DATA', 'train_dir')

landmark_detection = landmarkDetection(config)
sign_names = os.listdir(train_dir)
for sign_name in sign_names:
    for image in os.listdir(os.path.join(sign_names, sign_name):

        

