# Segments.ai upload 'Scott Base 2021-22' dataset

from os import environ, listdir, devnull
from segments import SegmentsClient
from tqdm import tqdm
from sys import stdout

api_key = environ.get("SegmentsAI_api_key")
dataset = "segmentsai1/Seal_2022-22"

client = SegmentsClient(api_key)
path = "/home/fdi19/SENG402/data/images/scott_base/2021-22"

for filename in tqdm(listdir(path)[8775:]):
    name = filename.split('.')[0]

    with open(f"{path}/{filename}", "rb") as f:
        asset = client.upload_asset(f, filename=filename)

    attributes = {"image": {"url": asset["url"]}}
    client.add_sample(dataset, name, attributes)
