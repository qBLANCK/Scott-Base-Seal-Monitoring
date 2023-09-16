#!/bin/bash

# Note: The number of chunks generated and the length of the heatmap decay can be checked in create_heatmap.py

# Specify the number of video clips (must match the number of chunks in the heatmap_chunks folder)
num_clips=32

# Specify the durations for each clip (should match the length of each chunk minus HEATMAP_HEAT_DECAY)
# The first chunk is one second shorter than the rest
durations=(10.6 $(for ((i=1; i<num_clips; i++)); do echo 11.6; done))

prefix="heatmap"

# Create the concat.txt file
input_file="concat.txt"
echo -n > "$input_file"

for ((i=1; i<=num_clips; i++)); do
    filename="${prefix}_${i:02}.mp4"
    echo "file '$filename'" >> "$input_file"
    echo "inpoint 00:00:00.000" >> "$input_file"
    echo "duration 00:00:${durations[i-1]:0:2}.${durations[i-1]:2:1}00" >> "$input_file"
done

# Concatenate the video clips
output_file="output.mp4"
ffmpeg -f concat -safe 0 -i "$input_file" -c:v libx264 -c:a aac -strict experimental -y "$output_file"

echo "Concatenation completed. Output saved as '$output_file'."
