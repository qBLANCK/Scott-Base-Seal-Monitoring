import csv
from pathlib import Path
from os import listdir
from pytz import country_names
import torch
import tools.image.cv as cv
from tqdm import tqdm
from checkpoint import load_model
from evaluate import evaluate_image
from detection import detection_table

# TODO: Take out dependence on library
# TODO: Add more confidence levels
# TODO: Add snowstorm detection


# CONSTANTS
MODEL_PATH = Path('log/Seals_2021-22/model.pth')
SEAL_IMG_DIR = Path("../../data/images/scott_base/2021-22")
OUTPUT_DIR = Path("../../data/counts")
OUTPUT_NAME = "scott_base-21-22.csv"
THRESHOLDS = [0.3, 0.4, 0.5]

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
                            label_class = classes[label]
                            seal_count += 1 if label_class.name == "seal" else 2

                    counts.append(seal_count)

                timestamp = seal_img_name.split(".")[0]
                count_writer.writerow([timestamp, *counts])
    except KeyboardInterrupt:
        count_file.close()
