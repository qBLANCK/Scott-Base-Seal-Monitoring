import csv
from tqdm import tqdm

INPUT_CSV = 'data/locations/2021-22_detection.csv'

OUTPUT_CSV = 'data/locations/2021-22_detection_filtered.csv'

# Function to calculate the Euclidean distance between two points
def calculate_distance(x1, y1, x2, y2):
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

# Read the CSV file
with open(INPUT_CSV, 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    header = next(csv_reader)  # Read and ignore the header row
    rows = list(csv_reader)

# Create a new list to store filtered rows
filtered_rows = []

# Create a tqdm progress bar
progress_bar = tqdm(total=len(rows), desc="Filtering seal locations")

# Loop through the rows to filter based on nearby detections
for i in range(len(rows)):
    x1, y1, time1 = map(int, rows[i])
    nearby_detection = False
    
    # Check adjacent timestamps (previous and next)
    for j in range(i - 1, i + 2):
        if j >= 0 and j < len(rows) and j != i:
            x2, y2, time2 = map(int, rows[j])
            distance = calculate_distance(x1, y1, x2, y2)
            
            # Check if the distance criteria is met
            if distance <= 50:
                nearby_detection = True
                break
    
    if nearby_detection:
        filtered_rows.append(rows[i])

    progress_bar.update(1)

progress_bar.close()

# Write the filtered rows to a new CSV file
print("Writing to CSV...")
with open(OUTPUT_CSV, 'w', newline='') as output_file:
    csv_writer = csv.writer(output_file)
    csv_writer.writerow(header)  # Write the header row
    csv_writer.writerows(filtered_rows)
