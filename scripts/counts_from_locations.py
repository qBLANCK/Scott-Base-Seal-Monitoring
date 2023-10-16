""" 
Generates a CSV file of seal counts from a CSV of seal locations with timestamps.
"""

import pandas as pd
import argparse

# Argument Parser
parser = argparse.ArgumentParser(description="Generate a CSV file of seal counts from seal locations with timestamps.")
parser.add_argument("--input", type=str, required=True, help="Input CSV file containing seal locations.")
parser.add_argument("--output", type=str, required=True, help="Output CSV file name and path.")
parser.add_argument("--thresholds", type=int, nargs='+', default=[50], help="List of confidence thresholds as percentages.")
args = parser.parse_args()

input_csv = args.input
output_csv = args.output
thresholds_percentages = args.thresholds

# Convert threshold percentages to decimal values
thresholds = [t / 100.0 for t in thresholds_percentages]

df = pd.read_csv(input_csv)

# Initialize a DataFrame to store counts for each threshold
counts_df = pd.DataFrame({"Timestamp": df["Timestamp"].unique()})

for threshold in thresholds:
    filtered_df = df[df["Confidence"] >= threshold] # Filter detections above the threshold
    
    # Group the data by the "Timestamp" column and count the occurrences
    counts = filtered_df.groupby("Timestamp").size().reset_index(name=f"Counts ({int(threshold*100)}%)")
    
    # Merge the counts with the existing DataFrame
    counts_df = pd.merge(counts_df, counts, on="Timestamp", how="left")

# Fill NaN values with 0 for timestamps with no detections above the thresholds
counts_df = counts_df.fillna(0)

output_csv = output_csv
counts_df.to_csv(output_csv, index=False)

print(f"Counts saved to {output_csv}")