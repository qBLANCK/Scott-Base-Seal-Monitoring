from math import floor
from random import shuffle

from tools import concat_lists


def tagged(tag, contents):
    return {'tag': tag, 'contents': contents}


def import_coco(annotations_dir, image_root, split_ratio):
    from pycocotools.coco import COCO

    coco = COCO(annotations_dir)

    cat_ids = coco.getCatIds()
    print("{} classes found".format(len(cat_ids)))

    cats = coco.loadCats(cat_ids)
    class_names = [cat['name'] for cat in cats]
    class_map = {
        cat['id']: {
            'name': cat['name'],
            'colour': hex(int('0x%02x%02x%02x' % tuple(cat['color']), 16)),
            'shape': 'box'
        } for cat in cats
    }

    image_ids = concat_lists([coco.getImgIds(catIds=[cat]) for cat in cat_ids])
    shuffle(image_ids)

    train_r, _, validate_n = split_ratio
    train_n = floor(len(image_ids) * train_r / 100)
    validate_n = floor(len(image_ids) * validate_n / 100)
    test_n = len(image_ids) - train_n - validate_n
    catagory = iter(concat_lists(
        [['train'] * train_n, ['test'] * test_n, ['validate'] * validate_n]))

    print("found images: ", len(image_ids))
    print(class_names)

    def convert(id):
        info = coco.loadImgs(id)[0]
        file_name = info['file_name']

        def import_ann(ann):
            x, y, w, h = ann['bbox']
            return {
                'label': ann['category_id'],
                'confirm': True,
                'detection': None,
                'shape': tagged('box', {'lower': [x, y], 'upper': [x + w, y + h]})
            }

        anns = coco.loadAnns(coco.getAnnIds(id, catIds=cat_ids))
        annotations = {k: import_ann(ann) for k, ann in enumerate(anns)}

        return {
            'image_file': f"{image_root}/{file_name}",
            'image_size': [info['width'], info['height']],
            'category': next(catagory),
            'annotations': annotations
        }

    images = list(map(convert, image_ids))

    return {
        'config': {
            'root': annotations_dir,
            'extensions': [".jpg"],
            'classes': class_map
        },
        'images': images
    }
