# Segments.ai create release from 'Scott Base 2021-22' dataset
# Note: Only labeled/reviewed images are included in the release

from os import environ
from segments import SegmentsClient, SegmentsDataset
from segments.utils import export_dataset
from time import sleep

api_key = environ.get("SegmentsAI_api_key")

dataset_identifier = "segmentsai1/Seal_2022-22"
name = "v1.2"


client = SegmentsClient(api_key)
# Create release
client.add_release(dataset_identifier, name)
# Initialize a SegmentsDataset from the release file
release = client.get_release(dataset_identifier, name)
while (status := release['status']) != 'SUCCEEDED':
    print(f"Dataset not uploaded, trying again in 5s.\nStatus: {status}")
    sleep(5)
    release = client.get_release(dataset_identifier, name)

dataset = SegmentsDataset(release, labelset='ground-truth',
                          filter_by=['labeled', 'reviewed'])

# Export to COCO instance format
export_dataset(dataset, export_format="coco-instance",
               export_folder="./data/annotations/")
