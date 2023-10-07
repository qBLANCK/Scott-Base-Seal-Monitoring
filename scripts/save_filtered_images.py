from PIL import Image, ImageDraw
import pandas as pd
import os

# Load the CSV file containing detection data
csv_path = 'data/locations/2021-22_locations_c50_filtered.csv'
df = pd.read_csv(csv_path)

images = [
    "2022-02-13T10_49_12.jpg",
    "2022-01-25T09_33_50.jpg",
    "2022-01-09T23_33_32.jpg",
    "2022-01-27T15_18_54.jpg",
    "2022-01-02T12_03_23.jpg",
    "2021-12-07T14_02_44.jpg",
    "2022-02-09T13_19_07.jpg",
    "2022-02-20T08_34_14.jpg",
    "2021-12-03T13_17_39.jpg",
    "2022-01-09T12_33_31.jpg",
    "2022-02-13T09_49_12.jpg",
    "2022-01-31T03_48_58.jpg",
    "2021-12-19T02_03_01.jpg",
    "2021-12-14T23_32_55.jpg",
    "2021-12-12T19_17_51.jpg",
    "2021-12-21T00_18_05.jpg",
    "2021-11-25T01_47_29.jpg",
    "2022-01-04T10_33_26.jpg",
    "2021-12-05T16_17_42.jpg",
    "2022-01-15T16_48_38.jpg"
]

scott_base_2021 = "/csse/research/antarctica_seals/images/scott_base/2021-22/"
image_indexes = []

for image in images:
    all_images = sorted(os.listdir(scott_base_2021))
    try:
        image_index = all_images.index(image)
        image_indexes.append(image_index)
        #print(f"Image: {image} corresponds to index: {image_index}")
    except ValueError:
        print(f'The image {image} was not found in the folder.')

output_dir = "/media/jte52/BLANCK/Seals/Evaluation"

i = 0
for image_name, image_index in zip(images, image_indexes):
    i += 1
    timestamp = 1000 * image_index * (1/24)

    df["Time (ms)"] = df["Time (ms)"].round(0)
    detections = df[df["Time (ms)"] == round(timestamp, 0)]

    count = 0
    for _, row in detections.iterrows():
        count += 1

    path = os.path.join(scott_base_2021, image_name)

    image = Image.open(path)
    draw = ImageDraw.Draw(image)

    count = 0
    for _, row in detections.iterrows():
        x1, y1 = row["X pos"], row["Y pos"]
        box = [(x1 - 5, y1 - 5), (x1 + 5, y1 + 5)]
        draw.rectangle(box, outline="red", width=2)
        count += 1
    print(i)
    print(f"Image: {image_name}")
    print(f"{count} seals detected in {image_name} with threshold 0.5 and adjacent image filtering\n")

    output_image_name = f"filtered_{image_name}"
    output_image_path = os.path.join(output_dir, output_image_name)
    image.save(output_image_path)