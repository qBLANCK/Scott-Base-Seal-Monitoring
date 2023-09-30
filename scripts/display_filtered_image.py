# 256417, 256375 and 256458 for seal day
from PIL import Image, ImageDraw
import pandas as pd
import math

# Load the CSV file containing detection data
csv_path = 'data/locations/2021-22_locations.csv'
df = pd.read_csv(csv_path)

# Specify the timestamps you want to compare (in milliseconds)
# timestamp1 = 256417  # Seal day
# timestamp2 = 256458  # Before seal day
# timestamp3 = 258375  # After seal day
timestamp1 = 123250.0
timestamp2 = 123208.0
timestamp3 = 123291.66666666666

# Filter the data for the specific timestamps
detections1 = df[df["Time (ms)"] == timestamp1]
detections2 = df[df["Time (ms)"] == timestamp2]
detections3 = df[df["Time (ms)"] == timestamp3]

# Load the image corresponding to timestamp1
#image_path = '/csse/research/antarctica_seals/images/scott_base/2021-22/2022-01-23T13_33_48.jpg'
image_path = '/csse/research/antarctica_seals/images/scott_base/2021-22/2021-12-21T05_18_05.jpg'
image = Image.open(image_path)

# Create a drawing context to overlay detections
draw = ImageDraw.Draw(image)

# Define the distance threshold for matching (e.g., 50 pixels)
distance_threshold = 25

validated_detections = 0
invalid_detections = 0

# Iterate through detections from timestamp1 and overlay them on the image
for _, row1 in detections1.iterrows():
    x1, y1 = row1["X pos"], row1["Y pos"]
    matched = False
    
    # Check if there is a matching detection in timestamp2
    for _, row2 in detections2.iterrows():
        x2, y2 = row2["X pos"], row2["Y pos"]
        print(x2)
        distance = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        
        if distance <= distance_threshold:
            matched = True
            break

    if matched == False:
        # Check if there is a matching detection in timestamp2
        for _, row3 in detections3.iterrows():
            x3, y3 = row3["X pos"], row3["Y pos"]
            distance = math.sqrt((x1 - x3) ** 2 + (y1 - y3) ** 2)
            
            if distance <= distance_threshold:
                matched = True
                break
    
    # Define the detection bounding box (adjust as needed)
    box = [(x1 - 5, y1 - 5), (x1 + 5, y1 + 5)]
    
    # Draw the bounding box in green if matched, otherwise in red
    outline_color = "green" if matched else "red"
    if matched: 
        validated_detections += 1
    else:
        invalid_detections += 1
    draw.rectangle(box, outline=outline_color, width=2)

# Iterate through detections from timestamp2 and overlay yellow circles
for _, row2 in detections2.iterrows():
    x2, y2 = row2["X pos"], row2["Y pos"]
    
    # Calculate the coordinates of the circle's bounding box
    circle_box = [(x2 - distance_threshold, y2 - distance_threshold),
                  (x2 + distance_threshold, y2 + distance_threshold)]
    
    # Draw the yellow circle
    outline_color = "yellow"
    draw.ellipse(circle_box, outline=outline_color, width=2)

# Iterate through detections from timestamp3 and overlay orange circles
for _, row3 in detections3.iterrows():
    x3, y3 = row3["X pos"], row3["Y pos"]
    
    # Calculate the coordinates of the circle's bounding box
    circle_box = [(x3 - distance_threshold, y3 - distance_threshold),
                  (x3 + distance_threshold, y3 + distance_threshold)]
    
    # Draw the orange circle
    outline_color = "orange"
    draw.ellipse(circle_box, outline=outline_color, width=2)


# Save or display the image with overlays
# output_image_path = "overlayed_image.jpg"  # Replace with the desired output path
# image.save(output_image_path)

# If you want to display the image, you can uncomment the following line
print(f"Validated detections: {validated_detections}    Invalid detections: {invalid_detections}")
image.show()
