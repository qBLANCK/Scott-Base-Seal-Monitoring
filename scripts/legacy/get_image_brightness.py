import os
from libs.tools.image import cv
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import torch
from Models.Seals.mask.mask import load_mask

# Define your image directory
SEAL_IMG_DIR = Path("/csse/research/antarctica_seals/images/scott_base/2021-22/")
MASK_PATH = 'Models/Seals/mask/mask_2021-22.jpg'

# Get a list of image files
image_files = [
    os.path.join(SEAL_IMG_DIR, img)
    for img in os.listdir(SEAL_IMG_DIR)
    if img.endswith(".jpg")
]
image_files.sort()

mask = load_mask(MASK_PATH)

# Create lists to store timestamp and brightness values
timestamps = []
brightness_values = []

with tqdm(total=len(image_files), desc='Processing Images') as pbar:
    # Process each image
    for img_path in image_files:
        image = cv.imread_color(img_path)
        masked_brightness = torch.mean(image[mask == 1].float() / 255.0)

        brightness_values.append(masked_brightness.item())
        # Extract the timestamp from the image filename
        timestamp = os.path.splitext(os.path.basename(img_path))[0]
        timestamps.append(timestamp)
        pbar.update(1)

# Create a DataFrame with timestamp and brightness values
df = pd.DataFrame({'timestamp': timestamps, 'brightness': brightness_values})

# Save the DataFrame to a CSV file
df.to_csv('data/scott_base_brightness.csv', index=False)