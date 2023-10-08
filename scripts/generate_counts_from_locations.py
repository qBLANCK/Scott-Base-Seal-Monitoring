""" 
Generates a CSV file of seal counts from a CSV of seal locations with timestamps.
"""

import pandas as pd
from datetime import datetime, timedelta

locations_csv = "data/locations/2021-22_locations_c55_filtered.csv"
df = pd.read_csv(locations_csv)

# Group the data by the "Timestamp" column and count the occurrences
counts = df.groupby("Timestamp").size().reset_index(name="Counts (55)")
counts = counts.sort_values(by="Timestamp", ascending=True)
output_csv = "data/counts/2021-22_filtered_c55.csv"
counts.to_csv(output_csv, index=False)

output_csv = "data/counts/2021-22_filtered_c55.csv"
counts.to_csv(output_csv, index=False)
