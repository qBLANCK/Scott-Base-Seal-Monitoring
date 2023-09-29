"""
Detects seal locations in a set of images, saving the results to a CSV file.

Script Variables:
- MODEL_PATH: Path to the pre-trained detection model.
- MASK_PATH: Path to the mask image file used for excluding certain areas from detections.
- SEAL_IMG_DIR: Directory containing the input images.
- OUTPUT_DIR: Directory where the output files will be saved.
- OUTPUT_NAME: Name of the CSV file containing seal locations.
"""

import csv
import os
from pathlib import Path
import torch
from tqdm import tqdm

import libs.tools.image.cv as cv
from Models.Seals.checkpoint import load_model
from Models.Seals.detection import detection_table
from Models.Seals.evaluate import evaluate_image
from Models.Seals.mask.mask import load_mask

# CONSTANTS
MODEL_PATH = Path('Models/Seals/log/Dual_b4/model.pth')
MASK_PATH = 'Models/Seals/mask/mask_2021-22_ext.jpg'
SEAL_IMG_DIR = Path("/csse/research/antarctica_seals/images/scott_base/2021-22/")
OUTPUT_DIR = Path("data/locations")
OUTPUT_NAME = "2021-22_locations_c40.csv"

# MODEL SETUP
print("Status: Loading seal detection model")
model, encoder, args = load_model(MODEL_PATH)
device = torch.cuda.current_device()
model.to(device)
encoder.to(device)
classes = args.dataset.classes

def is_responsible_bbox(bbox, frame):
    x1, y1, x2, y2 = bbox
    h, w = abs(y2 - y1), abs(x2 - x1)
    # Area > 1000px
    area = h * w
    if area > 1000:
        return False
    # Point outside of frame
    if x1 < 0 or x2 > frame.shape[1] or y1 < 0 or y2 > frame.shape[0]:
        return False
    ratio = 5
    if (w / h) > ratio or (h / w) > ratio:
        return False
    return True

mask_matrix = load_mask(MASK_PATH)

# Method for creating a CSV with the seal detections and timestamps to be used for generating a heatmap.
TIMELAPSE_FPS =  24
def detect_seal_locations(model, encoder, device, image_files):
    """Use given model to detect seals and return list of points with their timecode.
       E.g. [(x-pos of seal, y-pos of seal, time (ms) of point)].
       Also save results to CSV for subsequent script runs."""
    print(f"Status: Detecting seal locations, saving results to {OUTPUT_NAME}")
    nms_params = detection_table.nms_defaults._extend(
        threshold=0.4)
    points = []
    with open(OUTPUT_DIR / OUTPUT_NAME, "w") as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(["X pos", "Y pos", "Time (ms)"])
        for i, img in enumerate(tqdm(image_files)):
            frame = cv.imread_color(img)
            results = evaluate_image(model, frame, encoder,
                                     nms_params=nms_params, device=device)
            img_points = [(int((x1 + x2) / 2), int((y1 + y2) / 2), 1000 * (i * (1 / TIMELAPSE_FPS))) for
                          x1, y1, x2, y2 in results.detections.bbox if is_responsible_bbox([x1, y1, x2, y2], frame)]

            for x, y, time_ms in img_points:
                if mask_matrix[y, x] == 0:  # Exclude points that are in the mask
                    writer.writerow([x, y, time_ms])
                    points.append((x, y, time_ms))

    return points

image_files = [
    os.path.join(SEAL_IMG_DIR, img)
    for img in os.listdir(SEAL_IMG_DIR)
    if img.endswith(".jpg")
]
image_files.sort()
points = detect_seal_locations(model, encoder, device, image_files)