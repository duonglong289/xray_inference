import torch


def nms(bboxes, iou_threshold=0.5):
    """
    Non-Maximum Suppression (NMS) for bounding boxes.
    Args:
        bboxes (torch.Tensor): Tensor of shape [num_objects, 6], where each row is [x_min, y_min, x_max, y_max, score, class_id].
        iou_threshold (float): IOU threshold for suppression.
    Returns:
        torch.Tensor: Tensor of filtered bounding boxes.
    """
    # Sort by score in descending order
    bboxes = bboxes[bboxes[:, 4].argsort(descending=True)]

    keep = []
    while bboxes.size(0) > 0:
        # Select the box with the highest score
        curr_box = bboxes[0]
        keep.append(curr_box)

        if bboxes.size(0) == 1:
            break

        # Compute IoU with the remaining boxes
        ious = compute_iou(curr_box[:4], bboxes[1:, :4])

        # Suppress boxes with IoU above the threshold
        bboxes = bboxes[1:][ious <= iou_threshold]

    return torch.stack(keep)

def compute_iou(box1, boxes):
    """
    Compute IoU between one box and a set of boxes.
    Args:
        box1 (torch.Tensor): Tensor of shape [4] (x_min, y_min, x_max, y_max).
        boxes (torch.Tensor): Tensor of shape [num_objects, 4].
    Returns:
        torch.Tensor: IoU values for the box1 with each box in `boxes`.
    """
    x1 = torch.max(box1[0], boxes[:, 0])
    y1 = torch.max(box1[1], boxes[:, 1])
    x2 = torch.min(box1[2], boxes[:, 2])
    y2 = torch.min(box1[3], boxes[:, 3])

    # Compute intersection
    inter_area = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    # Compute union
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = box1_area + boxes_area - inter_area

    return inter_area / union_area

def weighted_box_fusion(bboxes, iou_threshold=0.5):
    """
    Weighted Box Fusion (WBF) for bounding boxes.
    Args:
        bboxes (torch.Tensor): Tensor of shape [num_objects, 6], where each row is [x_min, y_min, x_max, y_max, score, class_id].
        iou_threshold (float): IOU threshold for fusion.
    Returns:
        torch.Tensor: Tensor of fused bounding boxes.
    """
    bboxes = bboxes[bboxes[:, 4].argsort(descending=True)]
    keep = []

    while bboxes.size(0) > 0:
        curr_box = bboxes[0]
        scores = [curr_box[4]]
        weight_sum = curr_box[4]
        x_min, y_min, x_max, y_max = curr_box[:4] * curr_box[4]

        ious = compute_iou(curr_box[:4], bboxes[1:, :4])
        overlapping = ious > iou_threshold

        for i, is_overlapping in enumerate(overlapping):
            if is_overlapping:
                overlap_box = bboxes[1:][i]
                weight = overlap_box[4]
                x_min += overlap_box[0] * weight
                y_min += overlap_box[1] * weight
                x_max += overlap_box[2] * weight
                y_max += overlap_box[3] * weight
                weight_sum += weight
                scores.append(overlap_box[4])

        x_min = x_min / weight_sum
        y_min = y_min / weight_sum
        x_max = x_max / weight_sum
        y_max = y_max / weight_sum
        score = sum(scores) / len(scores)

        keep.append(torch.tensor([x_min, y_min, x_max, y_max, score, curr_box[5]]))

        # Remove processed boxes
        bboxes = bboxes[1:][~overlapping]

    return torch.stack(keep)
