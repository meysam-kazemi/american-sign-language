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
    """Saves a list of landmarks to a specified pickle file."""
    with open(landmarks_dir, 'wb') as f:
        pickle.dump(all_landmarks, f)


def load_saved_landmarks(config):
    """
    Loads all landmark data and their corresponding labels from pickle files.

    Args:
        landmarks_dir (str): The directory where the .pkl landmark files are stored.

    Returns:
        tuple: A tuple containing two lists:
               - X (list): A list of all landmark data points.
               - y (list): A list of the corresponding string labels.
    """
    landmarks_dir = config.get("DATA", "landmarks_dir")
    all_landmarks_data = []
    all_labels = []

    print("\nLoading landmark data...")
    # Find all the pickle files in the specified directory
    pickle_files = [f for f in os.listdir(landmarks_dir) if f.endswith('.pkl')]

    for pkl_file in pickle_files:
        # The label is the filename without the '.pkl' extension
        label = os.path.splitext(pkl_file)[0]
        file_path = os.path.join(landmarks_dir, pkl_file)

        # Load the list of landmarks from the pickle file
        with open(file_path, 'rb') as f:
            landmarks = pickle.load(f)

        # Add the loaded data and corresponding labels to our main lists
        all_landmarks_data.extend(landmarks)
        all_labels.extend([label] * len(landmarks))
        print(f"  - Loaded {len(landmarks)} samples for label '{label}'")
    
    print("Data loading complete. ✔")
    return all_landmarks_data, all_labels



def main():
    """
    This function is for finding landmarks from train dataset and save them.
    """
    landmark_detection = landmarkDetection(config)
    all_landmarks = []
    sign_names = os.listdir(TRAIN_DIR)
    os.makedirs(LANDMARKS_DIR, exist_ok=True)
    for sign_name in sign_names:
        if sign_name=="nothing":
            continue
        #os.makedirs("dataset/with_landmarks/"+sign_name, exist_ok=True)
        image_paths = os.listdir(os.path.join(TRAIN_DIR, sign_name))
        for i, image_path in enumerate(image_paths):
            img = cv.imread(os.path.join(TRAIN_DIR, sign_name, image_path), 1)
            img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            landmarks = landmark_detection.detect(img_rgb)
            #landmark_detection.draw_with_landmark(img, landmarks.multi_hand_landmarks)
            if landmarks.multi_hand_landmarks:
                all_landmarks.append(landmarks.multi_hand_landmarks)
                #cv.imwrite(f"dataset/with_landmarks/{sign_name}/{str(i)}.png", img)
            else:
                continue
            loading = (50*i/len(image_paths)) + 1
            print(f'{sign_name:<5}:[{"="*math.ceil(loading):<50}] {loading:.2f}%', end='\r')
        print(f'\nSaving {sign_name} in {LANDMARKS_DIR+sign_name+".pkl"} ...', end='\r')
        print(f'Saved {sign_name} in {LANDMARKS_DIR+sign_name+".pkl"} ✔')
        save_landmarks(all_landmarks, landmarks_dir=LANDMARKS_DIR+sign_name+".pkl")

if __name__=="__main__":
    main()

