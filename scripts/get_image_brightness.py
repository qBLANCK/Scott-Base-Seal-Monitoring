import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

# Define your image directory
SEAL_IMG_DIR = Path("/csse/research/antarctica_seals/images/scott_base/2021-22/")

# Get a list of image files
image_files = [
    os.path.join(SEAL_IMG_DIR, img)
    for img in os.listdir(SEAL_IMG_DIR)
    if img.endswith(".jpg")
]
image_files.sort()

# Create lists to store timestamp and brightness values
timestamps = []
brightness_values = []

with tqdm(total=len(image_files), desc='Processing Images') as pbar:
    # Process each image
    for img_path in image_files:
        image = cv2.imread(img_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Calculate the mean brightness and normalize it (if needed)
        mean_brightness = gray_image.mean() / 255.0  # Normalize to [0, 1] range
        brightness_values.append(mean_brightness)
        # Extract the timestamp from the image filename
        timestamp = os.path.splitext(os.path.basename(img_path))[0]
        timestamps.append(timestamp)
        pbar.update(1)

# Create a DataFrame with timestamp and brightness values
df = pd.DataFrame({'timestamp': timestamps, 'brightness': brightness_values})

# Save the DataFrame to a CSV file
df.to_csv('data/brightness_values.csv', index=False)
