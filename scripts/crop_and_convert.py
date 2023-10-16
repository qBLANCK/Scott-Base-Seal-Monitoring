"""
This script converts .RW2 files to .JPG format, crops them to a specified region, and renames them based on their timestamp.

Some values from previous datasets that may be useful:
    Crop box for 2021-22 dataset: (4600, 7250, 8514, 7625)  - Images scaled by 2 after cropping
    Crop box for 2022-23 dataset: (6000, 6900, 15750, 7800)
"""

import os
import rawpy
from PIL import Image
from tqdm import tqdm
import exifread
import datetime
import argparse

# Argument Parser
parser = argparse.ArgumentParser(description="Convert .RW2 files to .JPG format, crop, and rename based on timestamp.")
parser.add_argument("--input", type=str, required=True, help="Directory containing .RW2 files.")
parser.add_argument("--output", type=str, required=True, help="Directory to save the cropped .JPG files.")
parser.add_argument("--crop_box", type=int, nargs=4, required=True, help="Crop dimensions (left, upper, right, lower).")
parser.add_argument("--scale", type=float, default=1, required=True,  help="Scale factor to scale image after cropping")
args = parser.parse_args()

# Assign arguments to variables
input_dir = args.input
output_dir = args.output
crop_box = tuple(args.crop_box)
scale_factor = args.scale

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# List all .RW2 files in the input directory
rw2_files = [file for file in os.listdir(input_dir) if file.endswith('.RW2')]

progress_bar = tqdm(total=len(rw2_files), desc="Processing")

# Loop through each .RW2 file and convert & crop to .JPG
for rw2_file in rw2_files:
    input_path = os.path.join(input_dir, rw2_file)
    
    with rawpy.imread(input_path) as raw:
        rgb = raw.postprocess()

    with open(input_path, 'rb') as f:
        tags = exifread.process_file(f, details=False)
        timestamp_tag = tags.get('EXIF DateTimeOriginal', None)

    if timestamp_tag:
        # Rename file to timestamp
        timestamp = datetime.strptime(str(timestamp_tag), "%Y:%m:%d %H:%M:%S")
        new_filename = timestamp.strftime("%Y-%m-%dT%H_%M_%S") + '.jpg'

        output_path = os.path.join(output_dir, new_filename)

        # Process image
        image = Image.fromarray(rgb)
        image = image.crop(crop_box)
        
        # Scale the cropped image
        image = image.resize((image.width * scale_factor, image.height * scale_factor))

        image.save(output_path)
    else:
        print(f"Skipping {rw2_file}: EXIF metadata not found.")

    progress_bar.update(1)
progress_bar.close()

print("Conversion and cropping complete.")

