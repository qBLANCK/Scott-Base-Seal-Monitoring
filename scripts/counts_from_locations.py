""" 
Generates a CSV file of seal counts from a CSV of seal locations with timestamps.
"""

import pandas as pd

locations_csv = "data/locations/Locations_2021-22.csv"
df = pd.read_csv(locations_csv)
THRESHOLDS = [30, 40, 45, 50, 55, 60, 70]

# Initialize a dictionary to store counts for each threshold
counts_dict = {"Timestamp": df["Timestamp"].unique()}  # Initialize with unique timestamps

# Iterate over each confidence threshold
for threshold in THRESHOLDS:
    # Filter detections above the threshold
    filtered_df = df[df["Confidence"] >= threshold]
    
    # Group the data by the "Timestamp" column and count the occurrences
    counts = filtered_df.groupby("Timestamp").size().reset_index(name=f"Counts ({threshold})")
    
    # Merge the counts with the existing dictionary
    counts_dict = pd.merge(counts_dict, counts, on="Timestamp", how="left")

# Fill NaN values with 0 for timestamps with no detections above the thresholds
counts_dict = counts_dict.fillna(0)

output_csv = "data/counts/2021-22_filtered_counts.csv"
counts_dict.to_csv(output_csv, index=False)
