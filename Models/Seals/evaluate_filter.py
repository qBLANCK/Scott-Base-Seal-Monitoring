import csv

INPUT_CSV = 'scripts\heatmap\\2021-22_detection_filtered.csv'
# Read the filtered CSV file
with open(INPUT_CSV, 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    header = next(csv_reader)  # Read and ignore the header row
    
    # Create a dictionary to store counts for each timestamp
    timestamp_counts = {}
    
    for row in csv_reader:
        timestamp = int(row[2])  # Assuming timestamp is in the third column
        if timestamp in timestamp_counts:
            timestamp_counts[timestamp] += 1
        else:
            timestamp_counts[timestamp] = 1

# Find the peak number of detections
max_count = max(timestamp_counts.values())

print("Peak number of detections:", max_count)
