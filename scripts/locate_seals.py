import csv
from os import listdir
import os
from pathlib import Path

import torch
from tqdm import tqdm

import libs.tools.image.cv as cv
from Models.Seals.checkpoint import load_model
from Models.Seals.checkpoint import load_model
from Models.Seals.detection import detection_table
from Models.Seals.evaluate import evaluate_image
from Models.Seals.mask.mask import load_mask
import numpy as np

# CONSTANTS
MODEL_PATH = Path('Models/Seals/log/Dual_b4/model.pth')
MASK_PATH = 'Models/Seals/mask/mask_2021-22_ext.jpg'
SEAL_IMG_DIR = Path("/csse/research/antarctica_seals/images/scott_base/2021-22/")
OUTPUT_DIR = Path("./data/locations")
OUTPUT_NAME = "Locations_2021-22.csv"
THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]
BRIGHTNESS_THRESHOLD = 0.6

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

def brighten_image_to_threshold(frame):
    frame = frame.float() / 255.0
    brightness = frame.mean()
    if brightness < BRIGHTNESS_THRESHOLD:
        scaling_factor = BRIGHTNESS_THRESHOLD / brightness
        frame = torch.clamp(frame * scaling_factor, 0, 1)
    frame = (frame * 255).byte()
    return frame, brightness

mask_matrix = load_mask(MASK_PATH)

image_files = [
    os.path.join(SEAL_IMG_DIR, img)
    for img in os.listdir(SEAL_IMG_DIR)
    if img.endswith(".jpg")
]
image_files.sort()

# Write all detections above lowest threshold in THRESHOLDS to csv
with open(OUTPUT_DIR / OUTPUT_NAME, "w") as count_file:
    try:
        csv_writer = csv.writer(count_file, delimiter=',')
        csv_writer.writerow(["Timestamp", "X_min", "Y_min", "X_max", "Y_max", "Confidence"])

        for image_name in tqdm(image_files):
            frame = cv.imread_color(image_name)
            frame, brightness = brighten_image_to_threshold(frame)

            nms_params = detection_table.nms_defaults._extend(threshold = THRESHOLDS[0])
            results = evaluate_image(model, frame, encoder, nms_params = nms_params, device=device)

            d, p = results.detections, results.prediction
            
            detections = list(zip(d.label, d.bbox, d.confidence))

            for label, bbox, confidence in detections:
                if is_responsible_bbox(bbox, frame):
                    # Convert bbox coordinates to integers and create a mask for the bounding box
                    x_min, y_min, x_max, y_max = map(int, bbox)
                    bbox_mask = mask_matrix[y_min:y_max, x_min:x_max]
                    if not torch.any(bbox_mask):
                        timestamp = os.path.basename(image_name).split(".")[0]
                        csv_writer.writerow([timestamp, x_min, y_min, x_max, y_max, round(confidence.item(), 3)])

    except KeyboardInterrupt:
        count_file.close()