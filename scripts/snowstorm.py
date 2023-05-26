import torch
import csv
from tqdm import tqdm
from Models.Snowstorm.helper import load_model, classify


device = torch.cuda.current_device()
model = load_model("Models/Snowstorm/storm_model.pt")
model.to(device)

with open('data/counts/scott_base-21-22.csv','r') as csvinput:
    with open('output.csv', 'w') as csvoutput:
        writer = csv.writer(csvoutput, lineterminator='\n')
        reader = csv.reader(csvinput)

        all = []
        row = next(reader)
        row.append('Snowstorm')
        row.append('Snowstorm Confidence')
        all.append(row)

        for row in reader:
            path = f"/home/jte52/SENG402/data/images/scott_base/2021-22/{row[0]}.jpg"
            output, confidence = classify(model, device, path)

            row.append("Yes" if output == "storm" else "No")
            row.append(f"{confidence:.2f}")
            all.append(row)

        writer.writerows(all)
