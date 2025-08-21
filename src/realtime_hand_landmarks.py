import cv2
import mediapipe as mp
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.utils import load_config

class landmarkDetection:
    def __init__(self, config):
        self.min_confidence = float(config.get("LANDMARK_DETECTION", 'min_confidence'))
        self.num_hands = int(config.get("LANDMARK_DETECTION", 'num_hands'))

        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=self.num_hands,
            min_detection_confidence=self.min_confidence,
        )

    def detect(self, image):
        return self.hands.process(image)
    
    def draw_with_landmark(self, image, multi_hand_landmarks=None):
        if multi_hand_landmarks: # if the hand be in the image.
            # Loop through all detected hands
            for hand_landmarks in multi_hand_landmarks:
                # Draw the landmarks and their connections on the frame
                self.mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(255, 22, 76), thickness=2, circle_radius=4), # Landmark style
                    self.mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2)  # Connection style
                )

    def draw_without_landmark(self, image):
        res = self.detect(image)
        multi_hand_landmarks = res.multi_hand_landmarks
        self.draw_with_landmark(image, multi_hand_landmarks)
        return res



def main():
    """
    Main function to capture webcam feed and display hand landmarks.
    """

    config = load_config()

    landmard_detection = landmarkDetection(config)
    # Start capturing video from the webcam (device 0)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Webcam opened successfully. Press 'q' to quit.")

    while cap.isOpened():
        # Read a frame from the webcam
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Flip the frame horizontally for a more intuitive, mirror-like display
        frame = cv2.flip(frame, 1)

        # Convert the BGR image to RGB, as MediaPipe requires RGB input
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the image and find hand landmarks
        landmark = landmard_detection.detect(image_rgb)
        print(landmark.multi_hand_landmarks)
        landmard_detection.draw_with_landmark(frame, landmark.multi_hand_landmarks)

        # Display the frame in a window
        cv2.imshow('MediaPipe Hand Landmarks', frame)

        # Break the loop if the 'q' key is pressed
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    # Release resources
    hands.close()
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam feed closed.")

if __name__ == '__main__':
    main()

