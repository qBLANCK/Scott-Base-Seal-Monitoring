import os
from libs.tools.image import cv
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import torch
from Models.Seals.mask.mask import load_mask
import argparse

# Argument Parser
parser = argparse.ArgumentParser(description="Update counts CSV with snowstorm detections")
#parser.add_argument("--mask", type=str, required=True, help="Path to the mask image.")
#parser.add_argument("--input_csv", type=str, required=True, help="Seal counts CSV")
#parser.add_argument("--input_dir", type=str, required=True, help="Directory containing seal images.")
parser.add_argument("--output", type=str, required=True, help="Where to save the output CSV file.")
parser.add_argument("--brightness", type=float, default=0.6, help="Brightness threshold to flag as snowstorm.")
parser.add_argument("--confidence", type=float, default=0.5, help="Seal detection confidence threshold to use")
args = parser.parse_args()

#seal_img_dir = Path(args.input_dir)
#counts_csv = args.input_csv
#mask_path = args.mask
brightness_threshold = args.brightness
confidence_threshold = args.confidence

counts_csv = "data/counts/Counts_2021-22.csv"
seal_img_dir = Path("/csse/research/antarctica_seals/images/scott_base/2021-22/")
mask_path = 'Models/Seals/mask/mask_2021-22.jpg'

# Get a list of image files
image_files = [
    os.path.join(seal_img_dir, img)
    for img in os.listdir(seal_img_dir)
    if img.endswith(".jpg")
]
image_files.sort()

mask = load_mask(mask_path)

# Create lists to store timestamp and brightness values
snowstorms = []
brightness_values = []
timestamps = []

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

print("Filling in missing timestamps...")
brightness_df = pd.DataFrame({'Timestamp': timestamps, 'Brightness': brightness_values})
timestamps = set(timestamps)

counts_df = pd.read_csv(counts_csv)
merged_df = pd.merge(counts_df, brightness_df, on='Timestamp', how='outer')
merged_df = merged_df.fillna(0)

print("Checking for snowstorms...")

merged_df['Snowstorm'] = False
merged_df['Snowstorm'] = 'no'  # Initialize all entries to 'no'



# If brightness threshold is low but count is 0, set to "maybe"
merged_df.loc[
    (merged_df['Brightness'] <= brightness_threshold) &
    (merged_df['Counts ({}%)'.format(int(confidence_threshold * 100))] == 0),
    'Snowstorm'
] = 'maybe'

# If brightness threshold is high but count is below 10, set to "maybe"
merged_df.loc[
    (merged_df['Brightness'] > brightness_threshold) &
    (merged_df['Counts ({}%)'.format(int(confidence_threshold * 100))] < 10),
    'Snowstorm'
] = 'maybe'

# Check for "yes" and "maybe" cases
# If brightness is above the threshold and count is 0, set to "yes"
merged_df.loc[
    (merged_df['Brightness'] > brightness_threshold) &
    (merged_df['Counts ({}%)'.format(int(confidence_threshold * 100))] == 0),
    'Snowstorm'
] = 'yes'

merged_df = merged_df.sort_values(by='Timestamp')

print("Writing to CSV...")
merged_df.to_csv(args.output, index=False)
print("Done.")