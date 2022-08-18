import torch
from Models.Seals.checkpoint import load_model
from libs.tools import struct

import Models.Seals.evaluate as evaluate
from Models.Seals.dataset.imports import load_dataset

out_dir = "./Models/Seals/log/Seals"
model, encoder, args = load_model(f"{out_dir}/model.pth")
device = torch.cuda.current_device()
model = model.to(device)
encoder = encoder.to(device)
classes = args.dataset.classes

# scott_base_2021 = "/home/fdi19/SENG402/data/images/scott_base/2021-22/"
# image = random.choice(os.listdir(scott_base_2021))
# path = os.path.join(scott_base_2021, image)
# frame = cv.imread_color(path)

nms_params = struct(
    nms=0.5,
    threshold=0.5,
    detections=500)


eval_params = struct(
    overlap=200,
    split=True,
    image_size=(600, 600),
    batch_size=8,
    nms_params=nms_params,
    device=device,
    debug=None
)

input_params = struct(input=struct(choice='coco', parameters=struct(path='/home/fdi19/SENG402/data/annotations/export_coco-instance_segmentsai1_Seal_2022-22_v1.1.json',
                      image_root='/home/fdi19/SENG402/data/images/scott_base/2021-22', split_ratio='70/0/30')))
eval_test = evaluate.eval_test(model.eval(), encoder, eval_params)
_, dataset = load_dataset(input_params)
images = dataset.validate_images
loader = iter(dataset.test_on(images, struct(
    augment=None, num_workers=4), encoder))
result = eval_test(next(loader))
print(result.detections.bbox)
