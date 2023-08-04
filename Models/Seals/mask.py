from pathlib import Path
import torch
from matplotlib import pyplot as plt
import cv2
import numpy as np

''' Load a mask image and return it as a matrix of 0s and 1s. '''
''' The mask image should be an image containing only black and white pixels, equal to the size of the dataset '''
def load_mask(mask_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    _, mask_binary = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
    mask_matrix = np.where(mask_binary > 0, 1, 0)
    return torch.tensor(mask_matrix)

def display_mask(mask_matrix):
    plt.imshow(mask_matrix, cmap='gray', interpolation='nearest')
    plt.axis('off')
    plt.show()