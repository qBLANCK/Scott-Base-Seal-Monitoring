"""
Filters out detections that don't have a nearby detection in neighbouring timestamps.
Intended to work on files CSV created using generate_seal_locations.py

This script reads detections from a CSV file, filters out detections based on a distance threshold
to remove outliers, and writes the filtered detections to a new CSV file in the same format as the input.

distance_threshold can be modified to suit the dataset. 
This number should try to fit in the sweet spot between detecting everything as nearby and filtering out actually valid detections.
You may also consider modifying the number of timestamps either side of the timestamp you want to check, especially for sparser detections.
"""

import csv
import math
from tqdm import tqdm

# Define the path to your input CSV file
INPUT_CSV = 'data/locations/2021-22_locations_c55.csv'
OUTPUT_CSV = 'data/locations/2021-22_locations_c55_filtered.csv'

# Create a dictionary to store detections grouped by timestamp
detections_by_timestamp = {}

# Read the CSV file and group detections by timestamp
print("Status: Reading detections into dictionary")
with open(INPUT_CSV, 'r') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    for row in csv_reader:
        x_pos = float(row['X pos'])
        y_pos = float(row['Y pos'])
        timestamp = float(row['Time (ms)'])
        
        # Check if the timestamp exists in the dictionary, if not, create a new list
        if timestamp not in detections_by_timestamp:
            detections_by_timestamp[timestamp] = []
        
        # Append the detection to the corresponding timestamp
        detections_by_timestamp[timestamp].append((x_pos, y_pos))

# Define the distance threshold for matching
distance_threshold = 25

# Create a dictionary to store filtered detections by timestamp
filtered_detections_by_timestamp = {}

# Create a tqdm progress bar
progress_bar = tqdm(total=len(detections_by_timestamp), desc="Filtering detections")

# Iterate through the sorted dictionary of detections by timestamp
timestamps = sorted(detections_by_timestamp.keys())

for i, timestamp in enumerate(timestamps):
    filtered_detections = []
    
    for (x1, y1) in detections_by_timestamp[timestamp]:
        matched = False
        
        # Check for matching detections in adjacent timestamps
        for j in range(max(0, i - 1), min(i + 2, len(timestamps))):
            other_timestamp = timestamps[j]
            
            if other_timestamp == timestamp:
                continue  # Skip the same timestamp
            
            # Iterate through detections in the other timestamp
            for (x2, y2) in detections_by_timestamp[other_timestamp]:
                distance = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                
                if distance <= distance_threshold:
                    matched = True
                    break  # No need to check other detections in this timestamp
            
            if matched:
                break  # No need to check other timestamps if a match is found
        
        if matched:
            filtered_detections.append((x1, y1))

    filtered_detections_by_timestamp[timestamp] = filtered_detections
    progress_bar.update(1)
progress_bar.close()

# Write the filtered detections to a new CSV file
print("Status: Writing output to CSV")
with open(OUTPUT_CSV, 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    
    # Write the header row
    csv_writer.writerow(["X pos", "Y pos", "Time (ms)"])
    
    # Iterate through the filtered detections and write them to the CSV
    for timestamp, filtered_detections in sorted(filtered_detections_by_timestamp.items()):
        for (x, y) in filtered_detections:
            csv_writer.writerow([x, y, timestamp])
