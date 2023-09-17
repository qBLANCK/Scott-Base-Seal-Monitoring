import torch
import os

from libs.tools.image import cv
from Models.Seals.detection import display, detection_table
from Models.Seals.evaluate import evaluate_image
from Models.Seals.checkpoint import load_model
from Models.Seals.mask.mask import load_mask, apply_mask_to_image

# Load the model and other necessary components
out_dir = "Models/Seals/log/Seals02"
model, encoder, args = load_model(f"{out_dir}/model.pth")
device = torch.cuda.current_device()
model = model.to(device)
encoder = encoder.to(device)
classes = args.dataset.classes

# Load the mask and mask_extended (you can choose one)
MASK_PATH = "Models/Seals/mask/mask_2021-22_ext.jpg"
mask_t = load_mask(MASK_PATH)

def is_responsible_bbox(bbox, frame=None):
    x1, y1, x2, y2 = bbox
    h, w = abs(y2 - y1), abs(x2 - x1)
    # Area > 1000px
    area = h * w
    if area > 1000:
        return False
    # Point outside of frame
    if frame is not None:
        if x1 < 0 or x2 > frame.shape[1] or y1 < 0 or y2 > frame.shape[0]:
            return False
    ratio = 5
    if (w / h) > ratio or (h / w) > ratio:
        return False
    return True

def process_and_save_image(image_path, output_dir, threshold=0.5, mask=None, mask_visible=False):
    """
    Process an image, annotate it with seal detections, and save it to the specified directory.
    
    Args:
    image_path (str): Path to the input image.
    output_dir (str): Directory to save the processed image.
    threshold (float, optional): Detection threshold.
    mask (numpy.ndarray, optional): Mask image.
    mask_visible (bool, optional): Whether to display the mask on the image.

    Returns:
    None
    """
    frame = cv.imread_color(image_path)
    h, w = frame.shape[:2]
    nms_params = detection_table.nms_defaults._extend(threshold=threshold)
    results = evaluate_image(model, frame, encoder, nms_params=nms_params, device=device)

    d, p = results.detections, results.prediction

    detections = list(zip(d.label, d.bbox, d.confidence))

    if mask_visible:
        masked_frame = apply_mask_to_image(frame, mask)
        frame = torch.from_numpy(masked_frame)

    seal_count = 0
    for label, bbox, confidence in detections:
        if is_responsible_bbox(bbox, frame):
            label_class = classes[label]
            label_class.colour = "0xFF0000"

            if mask is not None:
                x_min, y_min, x_max, y_max = map(int, bbox)
                bbox_mask = mask[y_min:y_max, x_min:x_max]

                if not torch.any(bbox_mask):
                    seal_count += 1 if label_class.name == "seal" else 2
                    display.draw_box(frame, bbox, scale=1.0, color=display.to_rgb(label_class.colour), thickness=1)

            else:
                seal_count += 1 if label_class.name == "seal" else 2
                display.draw_box(frame, bbox, scale=1.0, color=display.to_rgb(label_class.colour), thickness=1)

    print(f"{seal_count} seals detected in {os.path.basename(image_path)} with threshold {threshold}{' using mask' if mask is not None else ''}")
    print("Image size (height, width):", h, "x", w)

    # Save the image with annotations
    output_path = os.path.join(output_dir, os.path.basename(image_path))
    cv.imwrite(output_path, frame)

# Images to process and save
image_paths = ['/home/jte52/images/2021-22/2022-01-23T13_33_48.jpg']
output_directory = 'data/processed_images/'
os.makedirs(output_directory, exist_ok=True)

for image_path in image_paths:
    process_and_save_image(image_path, output_directory, threshold=0.3, mask=mask_t, mask_visible=True)
