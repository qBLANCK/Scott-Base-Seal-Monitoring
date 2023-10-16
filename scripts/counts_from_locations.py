""" 
Generates a CSV file of seal counts from a CSV of seal locations with timestamps.
"""

import pandas as pd
import argparse

# Argument Parser
parser = argparse.ArgumentParser(description="Generate a CSV file of seal counts from seal locations with timestamps.")
parser.add_argument("--input", type=str, required=True, help="Input CSV file containing seal locations.")
parser.add_argument("--output", type=str, required=True, help="Output CSV file name and path.")
parser.add_argument("--thresholds", type=int, nargs='+', default=[30, 40, 50, 60, 70], help="List of confidence thresholds as percentages.")
args = parser.parse_args()

INPUT_CSV = args.input
OUTPUT_CSV = args.output
THRESHOLDS_PERCENTAGES = args.thresholds

# Convert threshold percentages to decimal values
THRESHOLDS = [t / 100.0 for t in THRESHOLDS_PERCENTAGES]

df = pd.read_csv(INPUT_CSV)

# Initialize a DataFrame to store counts for each threshold
counts_df = pd.DataFrame({"Timestamp": df["Timestamp"].unique()})

for threshold in THRESHOLDS:
    filtered_df = df[df["Confidence"] >= threshold] # Filter detections above the threshold
    
    # Group the data by the "Timestamp" column and count the occurrences
    counts = filtered_df.groupby("Timestamp").size().reset_index(name=f"Counts ({int(threshold*100)}%)")
    
    # Merge the counts with the existing DataFrame
    counts_df = pd.merge(counts_df, counts, on="Timestamp", how="left")

# Fill NaN values with 0 for timestamps with no detections above the thresholds
counts_df = counts_df.fillna(0)

output_csv = OUTPUT_CSV
counts_df.to_csv(output_csv, index=False)

print(f"Counts saved to {output_csv}")
