# Preproccesing
# For now, only using 2018 - 2019
RESEARCH_DIR = '/csse/research/antarctica_seals/scott_base/2018-19/cropped'
CROPS_PER_IMG = 5
OUT_DIR = 'data/all'
CROP_SIZE = 500  # px

# Training
DATA_DIR = "./data"  # ImageFolder structure
# From [resnet, alexnet, vgg, squeezenet, densenet, inception]
MODEL_NAME = "resnet"
NUM_CLASSES = 2
BATCH_SIZE = 8
NUM_EPOCHS = 15
# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
FEATURE_EXTRACT = True
INPUT_SIZE = 224
