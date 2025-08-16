import cv2
import mediapipe as mp
import numpy as np

# --- MediaPipe Initialization ---
# Initialize the Hands solution
mp_hands = mp.solutions.hands
# Initialize the drawing utilities
mp_drawing = mp.solutions.drawing_utils

def main():
    """
    Main function to capture webcam feed and display hand landmarks.
    """
    # Initialize the Hands model
    # - static_image_mode=False: Treat the input images as a video stream.
    # - max_num_hands=2: Detect up to two hands.
    # - min_detection_confidence=0.5: Minimum confidence value for the hand detection to be considered successful.
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

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
        results = hands.process(image_rgb)

        # Check if any hands were detected
        if results.multi_hand_landmarks:
            # Loop through all detected hands
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw the landmarks and their connections on the frame
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4), # Landmark style
                    mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2)  # Connection style
                )
        
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

