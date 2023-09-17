from pathlib import Path
import torch
from matplotlib import pyplot as plt
import cv2
import numpy as np

def load_mask(mask_path):
    """ Load a mask image and return it as a matrix of 0s and 1s.
        The mask image should be an image containing only black and white pixels, equal to the size of the dataset """
    try:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Mask file not found: {mask_path}")
        
        _, mask_binary = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
        mask_matrix = np.where(mask_binary > 0, 1, 0)
        return torch.tensor(mask_matrix)
    except Exception as e:
        raise Exception(f"Error loading mask: {str(e)}")

def apply_mask_to_image(frame, mask):
    """ Applies a mask to an image by blending the masked area with purple. """
    frame_np = frame.numpy() if isinstance(frame, torch.Tensor) else frame

    masked_frame = frame_np.copy()

    alpha = 0.5
    colour = np.array([128, 0, 128]) # Purple
    masked_frame[mask == 1] = (1 - alpha) * frame_np[mask == 1] + alpha * colour

    return masked_frame

def display_mask(mask_matrix):
    """ Display a binary mask matrix as an image.   """
    plt.imshow(mask_matrix, cmap='gray', interpolation='nearest')
    plt.axis('off')
    plt.show()