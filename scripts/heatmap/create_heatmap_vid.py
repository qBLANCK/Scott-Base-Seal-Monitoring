import os
import csv
import numpy as np
from dateutil import parser
from moviepy.editor import TextClip, concatenate_videoclips, CompositeVideoClip, ImageSequenceClip
# CNN
import torch
from tqdm import tqdm
from Models.Seals.checkpoint import load_model
from Models.Seals.detection import detection_table
from Models.Seals.evaluate import evaluate_image
# Heatmap
from libs.heatmappy.heatmappy.heatmap import Heatmapper
from libs.heatmappy.heatmappy.video import VideoHeatmapper
from libs.tools.image import cv

# MoviePy and Heatmappy are insanely RAM/CPU hungry. I split creating timelapses into quarters to help.
# I also had to use Deeplearning01 https://wiki.canterbury.ac.nz/display/RCC/Deeplearning01+the+big+GPU+machine
CHUNKS = 4
CHUNK_N = 0  # From 0 to CHUNK
MODEL_DIR = "../Models/Seals/log/Seals_2021-22/model.pth"
DETECTION_CREATE_CSV = True
DETECTION_CSV_NAME = "2021-22_detection.csv"
DETECTION_THRESHOLD = 0.4
TIMELAPSE_INPUT = '/home/fdi19/SENG402/data/images/scott_base/2021-22'
TIMELAPSE_IMAGES = np.array(sorted(os.listdir(TIMELAPSE_INPUT)))
TIMELAPSE_IMAGES = [list(x) for x in np.array_split(
    TIMELAPSE_IMAGES, CHUNKS)][CHUNK_N]
TIMELAPSE_USE_EVERY = 1  # every nth frame
TIMELAPSE_FPS = 24
TIMELAPSE_NAME = f"timelapse_q{CHUNK_N+1}.mp4"
HEATMAP_FPS = 24
HEATMAP_NAME = f"heatmap_q{CHUNK_N+1}.mp4"
HEATMAP_BITRATE = "3000k"
HEATMAP_KEEP_HEAT = True
HEATMAP_HEAT_DECAY = 1  # Seconds
HEATMAP_POINT_DIAM = 40
HEATMAP_POINT_STRENGTH = 0.2
HEATMAP_POINT_OPACITY = 0.35


def is_responsible_bbox(bbox, frame):
    """Decides if givin bounding box is realistic in terms of area, ratio and point placement."""
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


def load_CNN_model():
    """Load CNN model into GPU. Returns (model, encoder, device)."""
    print("Status: Loading model")
    model, encoder, _ = load_model(MODEL_DIR)
    device = torch.cuda.current_device()
    return model.to(device), encoder.to(device), device


def create_timelapse(image_files):
    """Create timelapse video using list of image paths."""
    print("Status: Creating timelapse")
    # Local installation of ImageMagick. I ran into permission issues when trying to modify policy.xml
    os.environ["IMAGEMAGICK_BINARY"] = "/home/fdi19/ImageMagick-7.1.0/utilities/magick"
    clip_list = []
    for image_path in image_files:
        image_name = image_path.split("/")[-1]
        iso_datetime = image_name.split('.')[0]
        datetime = parser.parse(iso_datetime.replace("_", ":"))
        datetime_str = datetime.strftime("%d/%m/%Y %H:%M")
        txt_clip = TextClip(
            txt=datetime_str, fontsize=70, color='black').set_duration(1/TIMELAPSE_FPS).set_fps(TIMELAPSE_FPS)
        clip_list.append(txt_clip)
    timestamps = concatenate_videoclips(clip_list, method="compose")
    timelapse = ImageSequenceClip(image_files, fps=TIMELAPSE_FPS)
    clip = CompositeVideoClip([timelapse, timestamps]).set_fps(TIMELAPSE_FPS)
    clip.write_videofile(TIMELAPSE_NAME, preset='slower', threads=16)


def detect_seals(model, encoder, device, image_files):
    """Use given model to detect seals and return list of points with their timecode.
       E.g. [(x-pos of seal, y-pos of seal, time (ms) of point)]."""
    print("Status: Detecting seals")
    nms_params = detection_table.nms_defaults._extend(
        threshold=DETECTION_THRESHOLD)
    points = []
    for i, img in tqdm(enumerate(image_files)):
        frame = cv.imread_color(img)
        results = evaluate_image(model, frame, encoder,
                                 nms_params=nms_params, device=device)
        img_points = [((x1 + x2) / 2, (y1 + y2) / 2, round(1000 * (i * (1 / TIMELAPSE_FPS)))) for x1, y1, x2,
                      y2 in
                      results.detections.bbox if is_responsible_bbox([x1, y1, x2, y2], frame)]
        points += img_points
    return points


def detect_seals_with_CSV(model, encoder, device, image_files):
    """Use given model to detect seals and return list of points with their timecode.
       E.g. [(x-pos of seal, y-pos of seal, time (ms) of point)].
       Also save results to CSV for subsequent script runs."""
    print("Status: Detecting seals, saving results to CSV")
    nms_params = detection_table.nms_defaults._extend(
        threshold=DETECTION_THRESHOLD)
    points = []
    with open(DETECTION_CSV_NAME, "w") as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(["X pos", "Y pos", "Time (ms)"])
        for i, img in tqdm(enumerate(image_files)):
            frame = cv.imread_color(img)
            results = evaluate_image(model, frame, encoder,
                                     nms_params=nms_params, device=device)
            img_points = [(int((x1 + x2) / 2), int((y1 + y2) / 2), round(1000 * (i * (1 / TIMELAPSE_FPS)))) for
                          x1, y1, x2,
                          y2 in results.detections.bbox if is_responsible_bbox([x1, y1, x2, y2], frame)]
            points += img_points
            writer.writerows(img_points)
    return points


def create_heatmap(points):
    """Create heatmap video given a list of points (x, y, time in ms)."""
    print(f"Status: Creating heatmap timelapse with {len(points)} points")
    img_heatmapper = Heatmapper(
        point_diameter=HEATMAP_POINT_DIAM, point_strength=HEATMAP_POINT_STRENGTH, opacity=HEATMAP_POINT_OPACITY)
    video_heatmapper = VideoHeatmapper(img_heatmapper)
    heatmap_video = video_heatmapper.heatmap_on_video_path(
        video_path=TIMELAPSE_NAME,
        points=points,
        keep_heat=HEATMAP_KEEP_HEAT,
        heat_decay_s=HEATMAP_HEAT_DECAY,
    )
    heatmap_video.duration = heatmap_video.end = heatmap_video.duration - HEATMAP_HEAT_DECAY
    heatmap_video.write_videofile(
        HEATMAP_NAME, bitrate=HEATMAP_BITRATE, fps=HEATMAP_FPS, threads=32)


if __name__ == "__main__":
    model, encoder, device = load_CNN_model()
    print("Status: Filtering images")
    image_files = [os.path.join(TIMELAPSE_INPUT, img)
                   for img in tqdm(TIMELAPSE_IMAGES[::TIMELAPSE_USE_EVERY])
                   if img.endswith(".jpg")]
    if os.path.exists(TIMELAPSE_NAME):
        print("Status: Timelapse already exists, skipping creation")
    else:
        create_timelapse(image_files)

    if DETECTION_CREATE_CSV and not os.path.exists(DETECTION_CSV_NAME):
        points = detect_seals_with_CSV(model, encoder, device, image_files)
    elif os.path.exists(DETECTION_CSV_NAME):
        print("Status: Reading points from csv")
        with open(DETECTION_CSV_NAME, newline='') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            timelapse_length = len(image_files) * (1 / TIMELAPSE_FPS) * 1000
            points = [(int(x), int(y), round(int(t) - CHUNK_N * timelapse_length))
                      for x, y, t in tqdm(list(reader))
                      if (int(t) - CHUNK_N * timelapse_length) < timelapse_length
                      and (int(t) - CHUNK_N * timelapse_length) > 0]
    create_heatmap(points)
