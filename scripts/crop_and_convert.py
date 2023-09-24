"""
This script converts .RW2 files to .JPG format, crops them to a specified region, and renames them based on their timestamp.
"""

import os
import rawpy
from PIL import Image
from tqdm import tqdm
import exifread
import datetime

# Directory containing .RW2 files
INPUT_DIR = '/media/jte52/Longterm_TL/2021-22_ANZ/Crater Hill 2022-2023/118_PANA'   # 2022-23 Dataset

# Directory to save the cropped .JPG files
OUTPUT_DIR = '/csse/research/antarctica_seals/images/scott_base/2022-23'

# Crop dimensions for 2022-23 dataset (left, upper, right, lower)
CROP_BOX = (6000, 6900, 15750, 7800)        # Crop dimensions (left, upper, right, lower)
#CROP_BOX = (4600, 7250, 8514, 7625)   # 2021-22 dataset


# Create the output directory if it doesn't exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# List all .RW2 files in the input directory
rw2_files = [file for file in os.listdir(INPUT_DIR) if file.endswith('.RW2')]

progress_bar = tqdm(total=len(rw2_files), desc="Processing")

# Loop through each .RW2 file and convert & crop to .JPG
for rw2_file in rw2_files:
    input_path = os.path.join(INPUT_DIR, rw2_file)
    
    with rawpy.imread(input_path) as raw:
        rgb = raw.postprocess()

    with open(input_path, 'rb') as f:
        tags = exifread.process_file(f, details=False)
        timestamp_tag = tags.get('EXIF DateTimeOriginal', None)

    if timestamp_tag:
        # Rename file to timestamp
        timestamp = datetime.strptime(str(timestamp_tag), "%Y:%m:%d %H:%M:%S")
        new_filename = timestamp.strftime("%Y-%m-%dT%H_%M_%S") + '.jpg'

        output_path = os.path.join(OUTPUT_DIR, new_filename)

        # Process image
        image = Image.fromarray(rgb)
        image = image.crop(CROP_BOX)
        
        # Scale the cropped image by a factor of 2 --- Uncomment if needed.
        #image = image.resize((image.width * 2, image.height * 2))

        image.save(output_path)
    else:
        print(f"Skipping {rw2_file}: EXIF metadata not found.")

    progress_bar.update(1)
progress_bar.close()

print("Conversion and cropping complete.")

