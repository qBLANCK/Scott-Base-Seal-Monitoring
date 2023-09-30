import csv

INPUT_CSV = 'data/locations/2021-22_locations_c60.csv'
# Read the filtered CSV file
with open(INPUT_CSV, 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    header = next(csv_reader)  # Read and ignore the header row
    
    # Create a dictionary to store counts for each timestamp
    timestamp_counts = {}
    
    for row in csv_reader:
        timestamp = round(float((row[2])))  # Assuming timestamp is in the third column
        if timestamp in timestamp_counts:
            timestamp_counts[timestamp] += 1
        else:
            timestamp_counts[timestamp] = 1

# Find the peak number of detections
max_count = max(timestamp_counts.values())

sorted_counts = sorted(timestamp_counts.items(), key=lambda x:x[1], reverse=True)

print(sorted_counts[:10])

from datetime import datetime, timedelta
start_time = datetime(2021, 11, 20, 10, 2) # Set start date and time (first image in dataset)

out = start_time + timedelta(milliseconds=256583 * 21600)   # Convert to datetime
print(f"Approx timestamp of highest counts: {out}")

print("Peak number of detections:", max_count)
