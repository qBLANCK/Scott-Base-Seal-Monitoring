"""
This script generates heatmaps from seal detection data and overlays them on a timelapse video.

        WARNING: THIS THING REQUIRES QUITE A LOT OF JUICE
        If you do not have access to a monstrously powerful PC such as DeepLearning01, 
        https://wiki.canterbury.ac.nz/display/RCC/Deeplearning01+the+big+GPU+machine
        consider looking into alternative heatmap generation methods.

        If you're getting really desperate, you can adjust NUM_CHUNKS until you get it to 
        generate one chunk then manually increment the loop everytime it crashes. 

- This script requires a CSV containing seal detections and a timelapse of the dataset. 
- These can be generated using generate_seal_locations.py and create_timelapse.py
- Place these files in the same directory and specify the paths in DETECTIONS_CSV and TIMELAPSE.
"""

import csv
import os
from tqdm import tqdm

import sys
sys.path.append('../../')  # Adjust path to access libs
from libs.heatmappy.heatmappy.heatmap import Heatmapper
from libs.heatmappy.heatmappy.video import VideoHeatmapper
from moviepy.editor import VideoFileClip

IMAGE_FOLDER = "/csse/research/antarctica_seals/images/scott_base/2021-22/"
# Adjust this to change the number of output chunks. Recommend making this as low as possible.
NUM_CHUNKS = 32  
# Path to CSV file containing seal detections. Generate this file using generate_seal_locations.py
DETECTIONS_CSV = "seal_locations.csv"   
FPS = 24
# Path to mp4 file containing timelapse of dataset. Generate this file using create_timelapse.py
TIMELAPSE = "2021-22_timelapse.mp4"   

# Heatmap parameters
HEATMAP_BITRATE = "3000k"
HEATMAP_KEEP_HEAT = True
HEATMAP_HEAT_DECAY = 1  # Seconds
HEATMAP_POINT_DIAM = 40
HEATMAP_POINT_STRENGTH = 0.5
HEATMAP_POINT_OPACITY = 0.35

# Get the list of image files in the folder
image_files = [os.path.join(IMAGE_FOLDER, f) for f in sorted(os.listdir(IMAGE_FOLDER)) if f.endswith(".jpg")]
total_frames = len(image_files)
frames_per_chunk = total_frames // NUM_CHUNKS

print("Status: Loading timelapse video")
timelapse_video = VideoFileClip(TIMELAPSE)

# Loop through each chunk
for i in range(int(NUM_CHUNKS)):
    print(f"Processing Chunk {i + 1}/{int(NUM_CHUNKS)}")

    # Calculate the time range for the current chunk
    start_frame = i * frames_per_chunk
    if i > 0:   # Add an overlap so that when stitching chunks together it looks seamless instead of 'resetting' the timelapse
        start_frame -= FPS * HEATMAP_HEAT_DECAY
    end_frame = (i + 1) * frames_per_chunk
    start_time = start_frame / FPS * 1000
    end_time = end_frame / FPS * 1000

    print("Status: Reading points from csv")
    with open(DETECTIONS_CSV, newline='') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        points = [(int(x), int(y), round(int(t) - start_time))
                    for x, y, t in tqdm(list(reader))
                    if start_time <= int(t) <= end_time]

    # Create heatmap video given a list of points (x, y, time in ms).
    # Don't touch this part if you don't need to
    print(f"Status: Creating heatmap timelapse with {len(points)} points")
    img_heatmapper = Heatmapper(
        point_diameter=HEATMAP_POINT_DIAM, point_strength=HEATMAP_POINT_STRENGTH, opacity=HEATMAP_POINT_OPACITY)
    video_heatmapper = VideoHeatmapper(img_heatmapper)
    heatmap_video = video_heatmapper.heatmap_on_video(
        base_video=timelapse_video.subclip(start_time/1000, end_time/1000),
        points=points,
        keep_heat=HEATMAP_KEEP_HEAT,
        heat_decay_s=HEATMAP_HEAT_DECAY,
    )
    heatmap_name = f"heatmap_chunks/heatmap_{i+1}.mp4"
    heatmap_video.duration = heatmap_video.end = heatmap_video.duration - HEATMAP_HEAT_DECAY
    heatmap_video.write_videofile(
        heatmap_name, bitrate=HEATMAP_BITRATE, fps=FPS, threads=32)

print("All chunks processed.")
