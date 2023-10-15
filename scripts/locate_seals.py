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
import numpy as np
import argparse

# CONSTANTS
DEFAULT_MODEL_PATH = 'Models/Seals/log/Dual_b4/model.pth'
DEFAULT_MASK_PATH = 'Models/Seals/mask/mask_2021-22_ext.jpg'
DEFAULT_SEAL_IMG_DIR = "/csse/research/antarctica_seals/images/scott_base/2021-22/"
DEFAULT_OUTPUT_NAME = "data/locations/Locations_2021-22_gen-Oct15.csv"
DEFAULT_CONFIDENCE_THRESHOLD = 0.3
DEFAULT_BRIGHTNESS_THRESHOLD = 0.6

FPS = 24

# Argument Parser
parser = argparse.ArgumentParser(description="Generate seal locations from images.")
parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH, help="Path to the seal detection model.")
parser.add_argument("--mask_path", type=str, default=DEFAULT_MASK_PATH, help="Path to the mask image.")
parser.add_argument("--input_dir", type=str, default=DEFAULT_SEAL_IMG_DIR, help="Directory containing seal images.")
parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_NAME, help="Where to save the output CSV file.")
parser.add_argument("--confidence_threshold", type=float, default=DEFAULT_CONFIDENCE_THRESHOLD, help="Confidence threshold for detections.")
parser.add_argument("--brightness_threshold", type=float, default=DEFAULT_BRIGHTNESS_THRESHOLD, help="Brightness threshold for images.")
args = parser.parse_args()

# MODEL SETUP
model_path = args.model_path
mask_path = args.mask_path
seal_img_dir = args.input_dir
output = args.output
confidence_threshold = args.confidence_threshold
brightness_threshold = args.brightness_threshold

print("Status: Loading seal detection model")
model, encoder, _ = load_model(model_path)
device = torch.cuda.current_device()
model.to(device)
encoder.to(device)

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

# # Does the whole image
# def brighten_image_to_threshold(frame):
#     frame = frame.float() / 255.0
#     brightness = frame.mean()
#     if brightness < brightness_threshold:
#         scaling_factor = brightness_threshold / brightness
#         frame = torch.clamp(frame * scaling_factor, 0, 1)
#     frame = (frame * 255).byte()
#     return frame, brightness

# Just checks the right half but still brightens everything
def brighten_image_to_threshold(frame):
    frame = frame.float() / 255.0
    
    # Calculate the mean brightness for the right half of the image
    width = frame.shape[1]
    right_half = frame[:, width // 2:]
    right_brightness = right_half.mean()
    
    if right_brightness < brightness_threshold:
        # If the right half is too dark, brighten the entire image
        scaling_factor = brightness_threshold / right_brightness
        frame = torch.clamp(frame * scaling_factor, 0, 1)
    
    frame = (frame * 255).byte()
    return frame, right_brightness

mask_matrix = load_mask(mask_path)

image_files = [
    os.path.join(seal_img_dir, img)
    for img in os.listdir(seal_img_dir)
    if img.endswith(".jpg")
]
image_files.sort()

# Write all detections above confidence threshold to csv
with open(output, "w") as count_file:
    try:
        csv_writer = csv.writer(count_file, delimiter=',')
        csv_writer.writerow(["Timestamp", "X_min", "Y_min", "X_max", "Y_max", "Confidence", "Timelapse_pos"])

        for i, image_name in enumerate(tqdm(image_files)):
            time_ms = 1000 * (i * (1 / FPS))        # Image position in timelapse in milliseconds

            frame = cv.imread_color(image_name)
            frame, brightness = brighten_image_to_threshold(frame)

            nms_params = detection_table.nms_defaults._extend(threshold = confidence_threshold)
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
                        csv_writer.writerow([timestamp, x_min, y_min, x_max, y_max, round(confidence.item(), 3), time_ms])

    except KeyboardInterrupt:
        count_file.close()