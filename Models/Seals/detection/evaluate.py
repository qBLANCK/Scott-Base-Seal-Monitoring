import torch

from Models.Seals.detection import box
from libs.tools import struct, const


def bookend(*xs, dim=0):
    def to_tensor(xs):
        tensor = xs if torch.is_tensor(xs) else torch.FloatTensor([xs])
        return tensor#.cpu()

    return torch.cat([to_tensor(x) for x in xs], dim)


def rev_cummax(v):
    flipped = v.flip(0)
    rev_max, _ = torch.cummax(flipped, dim=0)
    return rev_max.flip(0)


def area_under_curve(xs, ys):
    i = (xs[1:] != xs[:-1]).nonzero(as_tuple=False).squeeze(1)
    return ((xs[i + 1] - xs[i]) * ys[i + 1]).sum().item()


def compute_mAP(matches, confidence, num_target, weight=1, eps=1e-7):
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
    assert detections.label.dim() == 1 and target.label.dim() == 1
    n, m = detections._size, target._size

    if m == 0 or n == 0:
        return const(torch.FloatTensor(n).zero_())

    ious = box.iou_matrix(detections.bbox, target.bbox)
    return lambda threshold: _match_positives(
        detections.label, target.label, ious, threshold=threshold)


def mAP_classes(image_pairs, num_classes):
    confidence = torch.cat(
        [i.detections.confidence for i in image_pairs]).float()
    confidence, order = confidence.sort(0, descending=True)

    matchers = [match_positives(i.detections, i.target) for i in image_pairs]

    predicted_label = torch.cat(
        [i.detections.label for i in image_pairs])[order]
    target_label = torch.cat([i.target.label for i in image_pairs])

    num_targets = torch.bincount(target_label, minlength=num_classes)

    def f(threshold):
        matches = torch.cat([match(threshold).cpu() for match in matchers])[order.device]

        def compute_class(i):
            inds = [(predicted_label == i).nonzero(as_tuple=False).squeeze(1)]
            return compute_mAP(
                matches[inds], confidence[inds].cpu(), num_targets[i].item())

        return struct(
            total=compute_mAP(matches, confidence.cpu(),
                              target_label.size(0)),
            classes=[compute_class(i) for i in range(0, num_classes)]
        )

    return f
