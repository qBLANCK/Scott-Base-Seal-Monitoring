from PIL import Image, ImageDraw
import pandas as pd

# Load the CSV file containing detection data
csv_path = 'data/locations/2021-22_detection_filtered.csv'
df = pd.read_csv(csv_path)

# Specify the timestamp you want to overlay (in milliseconds)
target_timestamp = 256417  # Replace with the desired timestamp

# Filter the data for the specific timestamp
target_detections = df[df["Time (ms)"] == target_timestamp]

# Load the image corresponding to the timestamp
image_path = '/csse/research/antarctica_seals/images/scott_base/2021-22/2022-01-23T13_33_48.jpg'
image = Image.open(image_path)

# Create a drawing context to overlay detections
draw = ImageDraw.Draw(image)

# Iterate through the detections and overlay them on the image
count = 0
for index, row in target_detections.iterrows():
    count += 1
    x_pos, y_pos = row["X pos"], row["Y pos"]
    
    # Define the detection bounding box (adjust as needed)
    box = [(x_pos - 5, y_pos - 5), (x_pos + 5, y_pos + 5)]
    
    # Draw a rectangle around the detection
    draw.rectangle(box, outline="red", width=2)
print(count)

# Save or display the image with overlays
#output_image_path = "overlayed_image.jpg"  # Replace with the desired output path
#image.save(output_image_path)

# If you want to display the image, you can uncomment the following line
image.show()
