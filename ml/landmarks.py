"""Hand landmark detection and feature extraction using MediaPipe."""
import cv2
import numpy as np
import mediapipe as mp


class LandmarkDetector:
    """Detects hand landmarks and converts them to model-ready feature vectors."""

    NUM_LANDMARKS = 21  # MediaPipe hands returns 21 landmarks per hand

    def __init__(self, min_confidence=0.3, num_hands=2, static_image_mode=False):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.num_hands = num_hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=num_hands,
            min_detection_confidence=min_confidence,
        )

    @classmethod
    def from_settings(cls, settings, **kwargs):
        return cls(
            min_confidence=settings.landmark_min_confidence,
            num_hands=settings.landmark_num_hands,
            **kwargs,
        )

    def detect(self, image_rgb):
        """Run detection on an RGB image. Returns the raw MediaPipe result."""
        return self.hands.process(image_rgb)

    def detect_bgr(self, image_bgr):
        """Convenience wrapper that converts a BGR (OpenCV) frame first."""
        return self.detect(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))

    @staticmethod
    def to_feature_vector(hand_landmarks):
        """Convert one hand's landmarks into a translation/scale-normalized 63-vector.

        Coordinates are made relative to the wrist (landmark 0) and scaled by the
        largest coordinate span so the feature is robust to hand position and size.
        """
        coords = np.array(
            [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32
        )
        coords -= coords[0]  # translate so wrist is at the origin
        max_span = np.abs(coords).max()
        if max_span > 0:
            coords /= max_span
        return coords.flatten()

    def extract_features(self, image_rgb):
        """Detect and return a feature vector for the first detected hand, or None."""
        result = self.detect(image_rgb)
        if not result.multi_hand_landmarks:
            return None
        return self.to_feature_vector(result.multi_hand_landmarks[0])

    def draw(self, image_bgr, multi_hand_landmarks):
        """Draw landmarks/connections onto a BGR frame in place."""
        if not multi_hand_landmarks:
            return
        for hand_landmarks in multi_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                image_bgr,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(255, 22, 76), thickness=2, circle_radius=4),
                self.mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),
            )

    def close(self):
        self.hands.close()


def _demo():
    """Open the webcam and visualize detected landmarks. Press 'q' to quit."""
    from ml.config import get_settings

    detector = LandmarkDetector.from_settings(get_settings())
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    print("Webcam opened. Press 'q' to quit.")
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue
        frame = cv2.flip(frame, 1)
        result = detector.detect_bgr(frame)
        detector.draw(frame, result.multi_hand_landmarks)
        cv2.imshow("MediaPipe Hand Landmarks", frame)
        if cv2.waitKey(5) & 0xFF == ord("q"):
            break
    detector.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    _demo()
