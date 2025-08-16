import torch
import sys
import os
import cv2 as cv
from collections import deque
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.model_structure import CNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAINED_MODEL_PATH = './models/asl_cnn_model.pth'

model = CNN(num_classes=37)

if device==device==torch.device('cpu'):
    model.load_state_dict(torch.load(TRAINED_MODEL_PATH, weights_only=True, map_location=device))
else:
    model.load_state_dict(torch.load(TRAINED_MODEL_PATH, weights_only=True))


CLASS_MAP = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 
    9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 
    17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 
    25: 'Z', 26: '0', 27: '1', 28: '2', 29: '3', 30: '4', 31: '5', 32: '6', 
    33: '7', 34: '8', 35: '9', 36: 'Space'
}

model.eval()
model.to(device)

# --webcam--
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open Webcam.")
    exit()


# Image and ROI (Region of Interest) dimensions
IMG_SIZE = 50  # The input size for your CNN model
ROI_TOP, ROI_BOTTOM, ROI_RIGHT, ROI_LEFT = 100, 400, 350, 400

# Prediction smoothing parameters
PREDICTION_BUFFER_SIZE = 10  # Number of frames to average over
predictions_deque = deque(maxlen=PREDICTION_BUFFER_SIZE)
current_prediction = ""

# --- Main Loop ---
while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Flip the frame horizontally for a more intuitive mirror-like display
    frame = cv.flip(frame, 1)

    # Draw the ROI on the frame
    cv.rectangle(frame, (ROI_LEFT, ROI_TOP), (ROI_RIGHT, ROI_BOTTOM), (0, 255, 0), 2)

    print(frame.shape)
    # Extract the ROI
    roi = frame[ROI_TOP:ROI_BOTTOM, ROI_LEFT:ROI_RIGHT, :]

    # --- Preprocessing for the Model ---
    # 1. Convert to grayscale
    print(roi)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    # 2. Resize to the model's expected input size
    img_resized = cv.resize(frame, (IMG_SIZE, IMG_SIZE))

    # 3. Convert to a PyTorch tensor, normalize, and add dimensions
    #    The required shape is [batch_size, channels, height, width]
    img_tensor = torch.from_numpy(img_resized).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0)  # Add channel dimension: [50, 50] -> [1, 50, 50]
    img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension: [1, 50, 50] -> [1, 1, 50, 50]

    # Move tensor to the same device as the model
    img_tensor = img_tensor[0].to(device)
    img_tensor = img_tensor.permute(0, 3, 1, 2)

    # --- Prediction ---
    with torch.no_grad():  # Disable gradient calculation for inference
        output = model(img_tensor)
        # Get the index of the highest probability
        predicted_idx = torch.argmax(output, dim=1).item()
    
    # --- Smoothing and Display ---
    # Add the current prediction to the deque
    predictions_deque.append(predicted_idx)

    # If the deque is full, determine the most common prediction
    if len(predictions_deque) == PREDICTION_BUFFER_SIZE:
        most_common_idx = max(set(predictions_deque), key=predictions_deque.count)
        current_prediction = CLASS_MAP.get(most_common_idx, "Unknown")

    # Display the prediction on the screen
    cv.putText(frame, f"Prediction: {current_prediction}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv.LINE_AA)

    # Show the final frame
    cv.imshow('ASL Recognition', frame)

    # Exit the loop when 'q' is pressed
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# --- Cleanup ---
cap.release()
cv.destroyAllWindows()

