"""
Counts seals in a set of images and generates a CSV file containing the seal counts.

Script Variables:
- MODEL_PATH: Path to the pre-trained detection model.
- MASK_PATH: Path to the mask image file used for excluding certain areas from detections.
- SEAL_IMG_DIR: Directory containing the input images.
- OUTPUT_DIR: Directory where the output CSV file will be saved.
- OUTPUT_NAME: Name of the CSV file containing seal counts.
- THRESHOLDS: List of detection confidence thresholds to apply.
"""

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

# CONSTANTS
MODEL_PATH = Path('Models/Seals/log/Dual_b4/model.pth')
MASK_PATH = 'Models/Seals/mask/mask_2022-23.jpg'
SEAL_IMG_DIR = Path("/csse/research/antarctica_seals/images/scott_base/2022-23/")
OUTPUT_DIR = Path("./data/counts")
OUTPUT_NAME = "seal_counts_2022-23.csv"
THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]

# MODEL SETUP
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

with open(OUTPUT_DIR / OUTPUT_NAME, "w") as count_file:
    try:
        count_writer = csv.writer(count_file, delimiter=',')
        count_writer.writerow(
            ["Timestamp", "Count (t=30)", "Count (t=40)", "Count (t=50)", "Count (t=60)", "Count (t=70)"])

        # Sorted by timestamp
        for seal_img_name in tqdm(sorted(listdir(SEAL_IMG_DIR))):
            if seal_img_name.endswith(".jpg"):
                img_path = SEAL_IMG_DIR / seal_img_name
                frame = cv.imread_color(str(img_path))

                counts = []
                for t in THRESHOLDS:
                    nms_params = detection_table.nms_defaults._extend(
                        threshold=t)
                    results = evaluate_image(
                        model, frame, encoder, nms_params=nms_params, device=device)

                    d, p = results.detections, results.prediction
                    detections = list(zip(d.label, d.bbox))

                    seal_count = 0
                    for label, bbox in detections:
                        
                        if is_responsible_bbox(bbox, frame):
                            # Convert bbox coordinates to integers and create a mask for the bounding box
                            x_min, y_min, x_max, y_max = map(int, bbox)
                            bbox_mask = mask_matrix[y_min:y_max, x_min:x_max]

                            if not torch.any(bbox_mask):
                                        label_class = classes[label]
                                        seal_count += 1 if label_class.name == "seal" else 2

                    counts.append(seal_count)

                timestamp = seal_img_name.split(".")[0]
                count_writer.writerow([timestamp, *counts])
    except KeyboardInterrupt:
        count_file.close()