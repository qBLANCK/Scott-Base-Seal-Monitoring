import torch

from Models.Seals.detection import box
from libs.tools import struct, const

def bookend(*xs, dim=0):
    """ Concatenate a variable number of tensors along a specified dimension.   """
    def to_tensor(xs):
        tensor = xs if torch.is_tensor(xs) else torch.FloatTensor([xs])
        return tensor.cpu()

    return torch.cat([to_tensor(x) for x in xs], dim)

def rev_cummax(v):
    """ Compute the reverse cumulative maximum of a 1D tensor.  """
    flipped = v.flip(0)
    rev_max, _ = torch.cummax(flipped, dim=0)
    return rev_max.flip(0)

def area_under_curve(xs, ys):
    """ Compute the area under the curve (AUC) given two 1D tensors 
    representing the x and y coordinates of points. """
    i = (xs[1:] != xs[:-1]).nonzero(as_tuple=False).squeeze(1)
    return ((xs[i + 1] - xs[i]) * ys[i + 1]).sum().item()

def compute_mAP(matches, confidence, num_target, weight=1, eps=1e-7):
    """ Compute the mean Average Precision (mAP) given match results, 
    confidence scores, and the number of targets. """
    false_positives = ((1 - matches) * weight).cumsum(0)
    true_positives = (matches * weight).cumsum(0)

    recall = true_positives / (num_target if num_target > 0 else 1)
    precision = true_positives / \
        (true_positives + false_positives).clamp(min=eps)

    recall = bookend(0.0, recall, 1.0)
    precision = rev_cummax(bookend(1.0, precision, 0.0))

    false_positives = bookend(0, false_positives)
    true_positives = bookend(0, true_positives)

    return struct(
        recall=recall,
        precision=precision,

        confidence=bookend(1.0, confidence, 0.0),

        false_positives=false_positives,
        true_positives=true_positives,

        false_negatives=num_target - true_positives,
        n=num_target,

        mAP=area_under_curve(recall, precision))


def _match_positives(labels_pred, labels_target, ious, threshold=0.5):
    """ Compute binary matches between predicted labels and target labels based on IoU scores.  """
    n, m = labels_pred.size(0), labels_target.size(0)
    assert ious.size() == torch.Size([n, m])
    ious = ious.clone()

    matches = torch.FloatTensor(n).zero_()

    for i in range(0, n):
        iou, j = ious[i].max(0)
        iou = iou.item()

        if iou > threshold:
            # mark target overlaps to 0 so they won't be selected twice
            ious[:, j] = 0
            if labels_pred[i] == labels_target[j]:
                matches[i] = 1

    return matches


def match_positives(detections, target):
    """ Match positive instances between predicted detections and target instances based on IoU scores. 
    
        Returns a callable that can be used to match positive instances
        based on a specified IoU threshold. It first checks if there are valid detections
        and targets. If not, it returns a tensor of zeros. Otherwise, it computes the IoU
        matrix between detections and targets, and returns a function that matches positives
        based on the threshold when called.
    """
    assert detections.label.dim() == 1 and target.label.dim() == 1
    n, m = detections._size, target._size

    if m == 0 or n == 0:
        return const(torch.FloatTensor(n).zero_())

    ious = box.iou_matrix(detections.bbox, target.bbox)
    return lambda threshold: _match_positives(
        detections.label, target.label, ious, threshold=threshold)


def mAP_classes(image_pairs, num_classes):
    """ Compute mean Average Precision (mAP) for multiple classes.
    
        This function computes mAP for multiple classes based on detection results and target instances.
        It first calculates confidence scores and orders detections based on confidence. Then, it defines
        IoU matchers for each image pair. When called with a threshold, it computes mAP for each class and
        returns a structured result with total mAP and per-class mAP scores.
    """
    confidence = torch.cat(
        [i.detections.confidence for i in image_pairs]).float()
    confidence, order = confidence.sort(0, descending=True)

    matchers = [match_positives(i.detections, i.target) for i in image_pairs]

    predicted_label = torch.cat(
        [i.detections.label for i in image_pairs])[order]
    target_label = torch.cat([i.target.label for i in image_pairs])

    num_targets = torch.bincount(target_label, minlength=num_classes)

    def f(threshold):
        matches = torch.cat([match(threshold) for match in matchers]).to(order.device)[order]

        def compute_class(i):
            inds = [(predicted_label == i).nonzero(as_tuple=False).squeeze(1)]
            return compute_mAP(
                matches[inds], confidence[inds], num_targets[i].item())

        return struct(
            total=compute_mAP(matches, confidence,
                              target_label.size(0)),
            classes=[compute_class(i) for i in range(0, num_classes)]
        )

    return f
