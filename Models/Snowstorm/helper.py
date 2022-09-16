from matplotlib import pyplot as plt
import cv2
import random

from Models.Snowstorm.intervals import all_clears, all_storms
from Models.Snowstorm.constants import RESEARCH_DIR, CROPS_PER_IMG, OUT_DIR, CROP_SIZE


def leading_zeros(number, total_amount=4):
    count = total_amount - len(str(number))
    return '0' * count if count >= 0 else ''


def show(img, title='', scale=3, rgb_convert=True):
    if img is None:
        print(f'{title if title else "Image"} does not exist')
        return
    h, w = img.shape[:2]
    plt.figure(figsize=(w/h*scale, scale))
    plt.title(title)
    if rgb_convert:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)


def rand_crop(img, size):
    y, x = (random.randint(0, ax - size) for ax in img.shape[:2])
    cropped_image = img[y:y+size, x:x+size]
    return cropped_image


def create_and_save_crops(is_storm):
    data = all_storms if is_storm else all_clears
    total_count = 0
    for cam_id, storms in data.items():
        cam_count = 0  # Count of storm/clear images per Camera
        for prefix, intervals in storms:
            for start, end in intervals:
                assert start <= end
                cam_count += (end - start + 1)
                for img_id in range(start, end+1):
                    # For each image of a storm/clear
                    filename = f'{prefix}{leading_zeros(img_id)}{img_id}'
                    path = f'{RESEARCH_DIR}/Camera{cam_id}/{filename}.jpg'
                    img = cv2.imread(path)
                    if img is None:
                        continue
                    for i in range(CROPS_PER_IMG):
                        cropped_image = rand_crop(img, CROP_SIZE)
                        cv2.imwrite(
                            f'{OUT_DIR}/{"storm" if is_storm else "clear"}/{cam_id}_{filename}_{i+1}.jpg', cropped_image)
        total_count += (cam_count * CROPS_PER_IMG)

        print(f'{cam_count} images of storm/clear from Camera {cam_id}')
        print(f"{cam_count * CROPS_PER_IMG} total augmentations saved to '{OUT_DIR}/{'storm' if is_storm else 'clear'}'\n")
    print(f'Total augmentation count: {total_count}\n')
