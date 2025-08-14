import torch
import sys
import os
import cv2 as cv
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.model_structure import CNN

TRAINED_MODEL_PATH = './models/asl_cnn_model.pth'

model = CNN(num_classes=37)

model.load_state_dict(torch.load(TRAINED_MODEL_PATH, weights_only=True))


print(model)
