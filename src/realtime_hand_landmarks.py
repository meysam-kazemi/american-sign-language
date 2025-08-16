import cv2
import mediapipe as mp
import numpy as np

class landmarkDetection:
    def __init__(self, min_confidence=0.3, num_hands=2):
        self.min_confidence = min_confidence
        self.num_hands = num_hands

        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.3)

    def detect(self, image):
        return self.hands.process(image)
    
    def draw_with_landmark(self, image, multi_hand_landmarks=None):
        if multi_hand_landmarks: # if the hand be in the image.
            # Loop through all detected hands
            for hand_landmarks in multi_hand_landmarks:
                # Draw the landmarks and their connections on the frame
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(255, 22, 76), thickness=2, circle_radius=4), # Landmark style
                    mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2)  # Connection style
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

    landmard_detection = landmarkDetection()
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

