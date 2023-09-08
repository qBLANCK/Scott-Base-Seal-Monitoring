import cv2
import os
from datetime import datetime
from tqdm import tqdm

frame_folder = "/csse/research/antarctica_seals/images/scott_base/2021-22/" # Image sources
output_video = "2021-22_timelapse.mp4"
fps = 24  # Frames per second

# Get the list of image files in the folder
image_files = [os.path.join(frame_folder, f) for f in sorted(os.listdir(frame_folder)) if f.endswith(".jpg")]
#image_files = image_files[:100]        # Uncomment this if you want to just generate a quick test portion instead of the whole video
total_frames = len(image_files)

# Get the dimensions of the first image (assuming all images have the same dimensions)
# For 2021-22 dataset this should be 750x7828.
frame = cv2.imread(image_files[0])
height, width, layers = frame.shape

# Initialize the video writer
fourcc = cv2.VideoWriter_fourcc(*"H264")
out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

for image_file in tqdm(image_files, desc="Processing frames", unit="frame"):
    frame = cv2.imread(image_file)
 
    # Add datetime to frame in "DD/MM/YYYY HH:mm" format
    # This relies on the images being named in a certain format, for example: '2021-11-20T10_02_26.jpg'
    filename = os.path.basename(image_file)
    date_time_str = os.path.splitext(filename)[0].replace("_", ":").replace("T", " ")
    date_time_obj = datetime.strptime(date_time_str, "%Y-%m-%d %H:%M:%S")
    date_time_formatted = date_time_obj.strftime("%d/%m/%Y %H:%M")
    cv2.putText(frame, date_time_formatted, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), thickness=4)
 
    out.write(frame)

out.release()

print("Timelapse created successfully")
