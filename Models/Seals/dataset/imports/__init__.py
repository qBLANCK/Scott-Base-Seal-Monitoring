import json

from Models.Seals.dataset.annotate import decode_dataset
from libs.tools.parameters import get_choice
from .coco import import_coco


def import_json(filename):
    with open(filename, "r") as file:
        dataset = json.loads(file.read())
        # Convert keys of classes to integers (json objects are always string)
        config = dataset['config']
        config['classes'] = {int(k): v for k, v in config['classes'].items()}

        return dataset


def add_dict(d, k):
    d[k] = d[k] + 1 if k in d else 1
    return d


def summarise(images, classes):
    counts = sum([len(image['annotations']) for image in images])

    categories = {}
    for image in images:
        add_dict(categories, image['category'])

    annotated = categories.get(
        'train', 0) + categories.get('test', 0) + categories.get('validate', 0)

    print("using {:d} classes, found {:d} images, {:d} annotated with {:d} instances at {:.2f} per image"
          .format(len(classes), len(images), annotated, counts, counts / annotated))

    print(categories)


def import_dataset(input_args):
    choice, params = get_choice(input_args)

    if choice == 'json':
        print("loading json from: " + params.path)
        return import_json(params.path)
    elif choice == 'coco':
        print("loading coco from: " + params.path)
        ratio = tuple(map(int, params.split_ratio.split('/')))
        return import_coco(params.path, params.image_root, split_ratio=ratio)
    else:
        assert False, "unknown dataset type: " + choice

def load_dataset(args):
    dataset = import_dataset(args.input)

    if 'second_input' in args:
        second_dataset = import_dataset(args.second_input)
        dataset = combine_datasets(dataset, second_dataset)

    summarise(dataset['images'], dataset['config']['classes'])      # This seems to be spitting out the wrong information at the moment
    return decode_dataset(dataset)

def combine_datasets(first, second):               #NOTE: Check images marked for validation
    combined = {}

    # Combine the 'config' information              NOTE: Might not work with 2 json files
    combined['config'] = {
        'root': first['config']['root'],
        'extensions': first['config']['extensions'],
        'classes': first['config']['classes']
    }

    # Combine the 'images' information
    combined['images'] = first['images'] + second['images']

    return combined

