import csv
from os import listdir
import os
from pathlib import Path

import torch
from tqdm import tqdm

import libs.tools.image.cv as cv
from Models.Seals.checkpoint import load_model
from Models.Seals.detection import detection_table
from Models.Seals.evaluate import evaluate_image
import mask

# TODO: Take out dependence on library
# TODO: Add more confidence levels
# TODO: Add snowstorm detection


# CONSTANTS
MODEL_PATH = Path('Models/Seals/log/Dual_b4/model.pth')
MASK_PATH = 'Models/Seals/mask_extended.jpg'
SEAL_IMG_DIR = Path("/csse/research/antarctica_seals/images/scott_base/2021-22/")
OUTPUT_DIR = Path("./data/counts")
OUTPUT_NAME = "seal_counts.csv"
THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]
WANT_LOCATIONS = False # Set this to True if you want to generate a CSV with the location/timestamp of each detection

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

mask_matrix = mask.load_mask(MASK_PATH)

# Method for creating a CSV with the seal detections and timestamps to be used for generating a heatmap.
# Taken from create_heatmap_vid.py (detect_seals_with_CSV)and modified to include masking
TIMELAPSE_FPS =  24
def detect_seal_locations(model, encoder, device, image_files):
    """Use given model to detect seals and return list of points with their timecode.
       E.g. [(x-pos of seal, y-pos of seal, time (ms) of point)].
       Also save results to CSV for subsequent script runs."""
    print("Status: Detecting seal locations, saving results to CSV")
    nms_params = detection_table.nms_defaults._extend(
        threshold=0.5)
    points = []
    with open("seal_locations.csv", "w") as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(["X pos", "Y pos", "Time (ms)"])
        for i, img in tqdm(enumerate(image_files)):
            frame = cv.imread_color(img)
            results = evaluate_image(model, frame, encoder,
                                     nms_params=nms_params, device=device)
            img_points = [(int((x1 + x2) / 2), int((y1 + y2) / 2), round(1000 * (i * (1 / TIMELAPSE_FPS)))) for
                          x1, y1, x2,
                          y2 in results.detections.bbox if is_responsible_bbox([x1, y1, x2, y2], frame)]
            
            for x, y, time_ms in img_points:
                if mask_matrix[y, x] == 0:  # Exclude points that are in the mask
                    writer.writerow([x, y, time_ms])
                    points.append((x, y, time_ms))

    return points

if WANT_LOCATIONS:
    image_files = [
        os.path.join(SEAL_IMG_DIR, img)
        for img in tqdm(os.listdir(SEAL_IMG_DIR))
        if img.endswith(".jpg")
    ]
    points = detect_seal_locations(model, encoder, device, image_files)
else:
    with open(OUTPUT_DIR / OUTPUT_NAME, "w") as count_file:
        try:
            count_writer = csv.writer(count_file, delimiter=',')
            count_writer.writerow(
                ["Timestamp", "Count (t=30)", "Count (t=40)", "Count (t=50)"])

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