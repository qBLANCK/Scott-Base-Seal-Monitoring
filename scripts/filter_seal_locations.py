"""
Filters out detections that don't have a nearby detection in neighbouring timestamps.
Intended to work on files CSV created using locate_seals.py

This script reads detections from a CSV file, filters out detections based on a distance threshold
to remove outliers, and writes the filtered detections to a new CSV file in the same format as the input.
"""

import csv
import math
from tqdm import tqdm
import argparse

# Argument Parser
parser = argparse.ArgumentParser(description="Filter seal detections based on distance and confidence.")
parser.add_argument("--input", type=str, required=True, help="Path to the input CSV file with seal detections.")
parser.add_argument("--output", type=str, required=True, help="Path to the output filtered CSV file.")
parser.add_argument("--distance", type=float, default=25, help="Distance threshold for filtering detections.")
parser.add_argument("--confidence", type=float, default=0.3, help="Confidence threshold for filtering detections.")
parser.add_argument("--num_timestamps", type=int, default=1, help="Number of timestamps to check on either side.")
parser.add_argument("--num_to_verify", type=int, default=1, help="Number of adjacent detections required to verify an image")
args = parser.parse_args()

# Check if num_timestamps is not 0
if args.num_timestamps <= 0:
    raise argparse.ArgumentTypeError("num_timestamps cannot be less than 1")

# Check if num_to_verify is less than num_timestamps * 2
if args.num_to_verify >= args.num_timestamps * 2:
    raise argparse.ArgumentTypeError("num_to_verify must be less than num_timestamps * 2")

input_csv = args.input
output_csv = args.output
distance_threshold = args.distance
confidence_threshold = args.confidence
num_timestamps = args.num_timestamps
num_to_verify = args.num_to_verify

timestamps = []  # List of timestamps
detections_by_timestamp = {}  # Dictionary to store detections by timestamp

# Read the CSV file and group detections by timestamp
print("Status: Reading detections into dictionary")
with open(input_csv, 'r') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    for row in csv_reader:
        timestamp = row['Timestamp']
        x_min = int(row['X_min'])
        y_min = int(row['Y_min'])
        x_max = int(row['X_max'])
        y_max = int(row['Y_max'])
        confidence = float(row['Confidence'])
        time_ms = float(row['Timelapse_pos'])

        if confidence < confidence_threshold:
            continue
        
        # Store detections by timestamp
        if timestamp not in detections_by_timestamp:
            detections_by_timestamp[timestamp] = []
        detections_by_timestamp[timestamp].append((x_min, y_min, x_max, y_max, confidence, time_ms))
        if timestamp not in timestamps:
            timestamps.append(timestamp)

filtered_detections_by_timestamp = {}

progress_bar = tqdm(total=len(detections_by_timestamp), desc="Filtering detections")

# Iterate through the sorted dictionary of detections by timestamp
timestamps = sorted(detections_by_timestamp.keys())

for i, timestamp in enumerate(timestamps):
    filtered_detections = []
    
    for (x_min, y_min, x_max, y_max, confidence, time_ms) in detections_by_timestamp[timestamp]:
        # Calculate the center of the bounding box
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2

        matched_count = 0
        
        # Check for matching detections in adjacent timestamps
        for j in range(max(0, i - num_timestamps), min(i + num_timestamps + 1, len(timestamps))):
            other_timestamp = timestamps[j]
            
            if other_timestamp == timestamp:
                continue  # Skip the same timestamp
            
            # Iterate through detections in the other timestamp
            for (x_min2, y_min2, x_max2, y_max2, _, _) in detections_by_timestamp[other_timestamp]:
                # Calculate the center of the other bounding box
                center_x2 = (x_min2 + x_max2) / 2
                center_y2 = (y_min2 + y_max2) / 2

                # Calculate the Euclidean distance between centers
                distance = math.sqrt((center_x - center_x2) ** 2 + (center_y - center_y2) ** 2)

                
                if distance <= distance_threshold:
                    matched_count += 1
                    # If enough matches are found, break the loop
                    if matched_count >= num_to_verify:
                        break
            
            # If enough matches are found, break the loop
            if matched_count >= num_to_verify:
                break
        
        if matched_count >= num_to_verify:
            filtered_detections.append((x_min, y_min, x_max, y_max, confidence, time_ms))

    filtered_detections_by_timestamp[timestamp] = filtered_detections
    progress_bar.update(1)
progress_bar.close()

# Write the filtered detections to a new CSV file
print("Status: Writing output to CSV")
with open(output_csv, 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    
    # Write the header row
    csv_writer.writerow(["Timestamp", "X_min", "Y_min", "X_max", "Y_max", "Confidence", "Timelapse_pos"])
    
    # Iterate through the filtered detections and write them to the CSV
    for timestamp, filtered_detections in sorted(filtered_detections_by_timestamp.items()):
        for d in filtered_detections:
            csv_writer.writerow([timestamp, d[0], d[1], d[2], d[3], d[4], int(d[5])])
