import torchvision.ops as torchvision

from libs.tools import struct

nms_defaults = struct(
    nms=0.5,
    threshold=0.5,
    detections=500
)


def nms(prediction, params):
    inds = (prediction.confidence >= params.threshold).nonzero(
        as_tuple=False).squeeze(1)
    prediction = prediction._index_select(inds)._extend(index=inds)

    try:
        inds = torchvision.nms(
            prediction.bbox, prediction.confidence, params.nms)
    except RuntimeError:
        pass
    return prediction._index_select(inds)._take(params.detections)
