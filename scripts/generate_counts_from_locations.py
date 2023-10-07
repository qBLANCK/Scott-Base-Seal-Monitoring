""" 
Generates a CSV file of seal counts from a CSV of seal locations with timestamps.

IMPORTANT NOTE: The start_time variable should be set according to the timestamp of the first image in the corresponding dataset
"""

import pandas as pd
from datetime import datetime, timedelta

locations_csv = "data/locations/2021-22_locations_c55_filtered.csv"
df = pd.read_csv(locations_csv)

# Calculate the timestamps for 15-minute intervals
# Note: There is a slight drift in the real dataset timestamps, but it's extremely small so it's acceptable to round to 15 minutes.
df["Timestamp"] = df["Time (ms)"] * 21600   # Multiply milliseconds by 21600 to get 15-minute intervals

start_time = datetime(2021, 11, 20, 10, 2) # Set start date and time (first image in dataset)

df["Timestamp"] = start_time + df["Timestamp"].apply(lambda x: timedelta(milliseconds=x))   # Convert to datetime
df["Timestamp"] = df["Timestamp"].dt.strftime("%Y-%m-%dT%H_%M_%S")      # Reformat

# Group the data by the "Time (ms)" column and count the occurrences
counts = df["Timestamp"].value_counts().reset_index()
counts.columns = ["Timestamp", "Counts (55)"]             # '50' refers to the confidence interval used to generate the locations CSV
counts = counts.sort_values(by="Timestamp", ascending=True)

output_csv = "data/counts/2021-22_filtered_c55.csv"
counts.to_csv(output_csv, index=False)
