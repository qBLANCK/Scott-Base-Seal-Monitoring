from os import path

import torch

from Models.Seals.dataset.detection import DetectionDataset
from libs.tools import pluck, struct, table
from libs.tools import to_structs


def split_tagged(tagged):
    return tagged.tag, tagged.contents if 'contents' in tagged else None


def decode_obj(obj):
    tag, shape = split_tagged(obj.shape)

    if tag == 'box':
        return struct(
            label=obj.label,
            box=[*shape.lower, *shape.upper])

    elif tag == 'circle':
        x, y, r = *shape.centre, shape.radius

        return struct(
            label=obj.label, box=[x - r, y - r, x + r, y + r])
    else:
        # Ignore unsupported annotation for now
        return None


def lookup(mapping):
    def f(i):
        assert i in mapping, "missing key in mapping" + \
                             str(list(mapping.keys())) + ": " + str(i)
        return mapping[i]

    return f


def decode_object_map(annotations, config):
    mapping = class_mapping(config)

    objs = {k: decode_obj(a) for k, a in annotations.items()}
    objs = [ann._extend(id=int(k))
            for k, ann in objs.items() if ann is not None]

    boxes = pluck('box', objs)
    labels = list(map(lookup(mapping), pluck('label', objs)))

    ids = pluck('id', objs)

    return table(bbox=torch.FloatTensor(boxes) if len(boxes) else torch.FloatTensor(0, 4),
                 label=torch.LongTensor(labels),
                 id=torch.LongTensor(ids))


def class_mapping(config):
    return {int(k): i for i, k in enumerate(config.classes.keys())}


def decode_image(data, config):
    target = decode_object_map(data.annotations, config)

    return struct(
        id=data.image_file,
        file=path.join(config.root, data.image_file),
        target=target,
        category=data.category
    )


def decode_dataset(data):
    data = to_structs(data)
    config = data.config
    classes = [struct(id=int(k), **v) for k, v in config.classes.items()]

    images = {i.image_file: decode_image(i, config) for i in data.images}
    return config, DetectionDataset(classes=classes, images=images)
