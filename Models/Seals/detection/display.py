from libs.tools.image import cv


def to_rgb(_hex):
    _hex = int(_hex, 16)
    return ((_hex >> 16) & 255, (_hex >> 8) & 255, (_hex >> 0) & 255)


def draw_box(image, box, scale=1.0, name=None, confidence=None,
             thickness=2, color=(255, 0, 0), text_color=None):
    text_color = text_color or color
    image = cv.rectangle(
        image, box[:2], box[2:], color=color, thickness=int(thickness * scale))

    width = abs(box[0] - box[2])

    if not (name is None):
        image = cv.putText(
            image,
            name,
            (box[0],
             box[1] + width * 2),
            scale=0.7 * scale,
            color=text_color,
            thickness=int(
                1 * scale))

    if not (confidence is None):
        str = "{:.2f}".format(confidence)
        image = cv.putText(
            image,
            str,
            (box[0],
             box[3] - width * 2),
            scale=0.7 * scale,
            color=text_color,
            thickness=int(
                1 * scale))

    return image
