"""
This script generates heatmaps from seal detection data and overlays them on a timelapse video.

        WARNING: THIS THING REQUIRES QUITE A LOT OF JUICE
        If you do not have access to a monstrously powerful PC such as DeepLearning01, 
        https://wiki.canterbury.ac.nz/display/RCC/Deeplearning01+the+big+GPU+machine
        consider looking into alternative heatmap generation methods.

        If you're getting really desperate, you can adjust the number of chunks until you get it to 
        generate one chunk successfully and modify the loop manually to increment it as needed.

- This script requires a CSV containing seal detections and a timelapse of the dataset. 
- These can be generated using locate_seals.py and create_timelapse.py
- Place these files in the same directory and specify the path using arguments.
"""

import csv
import sys
sys.path.append('../../')  # Adjust path to access libs
from libs.heatmappy.heatmappy.heatmap import Heatmapper
from libs.heatmappy.heatmappy.video import VideoHeatmapper
from moviepy.editor import VideoFileClip
import argparse

# Argument Parser
parser = argparse.ArgumentParser(description="Create a heatmap using seal locations and a timelapse video")
parser.add_argument("--chunks", type=int, default=1, help="Number of chunks to split the heatmap into. Recommended to be as low as possible.")
parser.add_argument("--frames", type=int, required=True, help="Number of images in the dataset being processed.")
parser.add_argument("--timelapse", type=str, required=True, help="Path to timelapse video to process.")
parser.add_argument("--seals", required=True, help="Path to a CSV file containing seal detections")
parser.add_argument("--scale", type=float, default=1.0 , help="Scale factor to upsize or downsize images (e.g., 0.5 for half size, 2.0 for double size)")
args = parser.parse_args()

num_chunks = args.chunks
detections_csv = args.seals
timelapse = args.timelapse 
total_frames = args.frames
frames_per_chunk = total_frames // num_chunks
scale_factor = args.scale

# Heatmap parameters
FPS = 24
HEATMAP_BITRATE = "3000k"
HEATMAP_KEEP_HEAT = True
HEATMAP_HEAT_DECAY = 1  # Seconds
HEATMAP_POINT_DIAM = 40
HEATMAP_POINT_STRENGTH = 0.2
HEATMAP_POINT_OPACITY = 0.35

print("Status: Loading timelapse video")
timelapse_video = VideoFileClip(timelapse)

# Loop through each chunk
for i in range(int(num_chunks)):
    print(f"Processing Chunk {i + 1}/{int(num_chunks)}")

    # Calculate the time range for the current chunk
    start_frame = i * frames_per_chunk
    if i > 0:   # Add an overlap so that when stitching chunks together it looks seamless instead of 'resetting' the timelapse
        start_frame -= FPS * HEATMAP_HEAT_DECAY
    end_frame = (i + 1) * frames_per_chunk
    start_time = start_frame / FPS * 1000
    end_time = end_frame / FPS * 1000

    print("Status: Reading points from csv")
    with open(detections_csv, newline='') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        points = []  # Create an empty list to store (X_mid, Y_mid, time in ms)
        for row in reader:                
            _, x_min, y_min, x_max, y_max, _, time_ms = row
            time_ms = int(time_ms)  # Convert to integer

            # Check if the time_ms value falls within the current chunk's time frame
            if start_time <= time_ms <= end_time:
                x_mid = (int(x_min) + int(x_max)) // 2
                y_mid = (int(y_min) + int(y_max)) // 2
                # Adjust the time value to be relative to the start of the chunk
                time_ms -= start_time
                points.append((int(x_mid * scale_factor), int(y_mid * scale_factor), int(time_ms)))

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
        heatmap_name, bitrate=HEATMAP_BITRATE, fps=FPS, threads=64)

print("All chunks processed.")
