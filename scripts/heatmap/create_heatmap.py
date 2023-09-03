import csv
import os
from tqdm import tqdm

import sys
sys.path.append('../../')  # Adjust path to access libs
from libs.heatmappy.heatmappy.heatmap import Heatmapper
from libs.heatmappy.heatmappy.video import VideoHeatmapper
from moviepy.editor import VideoFileClip

IMAGE_FOLDER = "/csse/research/antarctica_seals/images/scott_base/2021-22/"
NUM_CHUNKS = 50  # Adjust the size of the chunk
#CHUNK = 0  # From 0 to (total_chunks - 1), initially set to 0
DETECTIONS_CSV = "seal_locations.csv"
FPS = 24
TIMELAPSE_NAME = "2021-22_timelapse.mp4"

HEATMAP_BITRATE = "3000k"
HEATMAP_KEEP_HEAT = True
HEATMAP_HEAT_DECAY = 0.5  # Seconds
HEATMAP_POINT_DIAM = 40
HEATMAP_POINT_STRENGTH = 0.5
HEATMAP_POINT_OPACITY = 0.6

# Get the list of image files in the folder
image_files = [os.path.join(IMAGE_FOLDER, f) for f in sorted(os.listdir(IMAGE_FOLDER)) if f.endswith(".jpg")]
total_frames = len(image_files)

# Load the timelapse video once
print("Status: Loading timelapse video")
timelapse_video = VideoFileClip(TIMELAPSE_NAME)

# Calculate the number of frames per chunk
frames_per_chunk = total_frames // NUM_CHUNKS

# Loop through each chunk
#for CHUNK_N in range(int(total_chunks)):
for i in range(1):
    print(f"Processing Chunk {i + 1}/{int(NUM_CHUNKS)}")

    # Calculate the time range for the current chunk
    start_frame = i * frames_per_chunk
    end_frame = (i + 1) * frames_per_chunk

    # Calculate the time range for the current chunk
    start_time = start_frame / FPS * 1000
    end_time = end_frame / FPS * 1000

    print("Status: Reading points from csv")
    with open(DETECTIONS_CSV, newline='') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        points = [(int(x), int(y), round(int(t) - start_time))
                    for x, y, t in tqdm(list(reader))
                    if start_time <= int(t) <= end_time]

    """Create heatmap video given a list of points (x, y, time in ms)."""
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
