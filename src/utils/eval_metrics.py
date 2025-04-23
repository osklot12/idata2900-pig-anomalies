import numpy as np

def voc_ap(rec, prec):
    """Compute VOC AP given precision and recall."""
    rec = np.concatenate(([0.0], rec, [1.0]))
    prec = np.concatenate(([0.0], prec, [0.0]))

    for i in range(prec.size - 1, 0, -1):
        prec[i - 1] = np.maximum(prec[i - 1], prec[i])

    indices = np.where(rec[1:] != rec[:-1])[0]
    ap = np.sum((rec[indices + 1] - rec[indices]) * prec[indices + 1])
    return ap

def compute_stats_from_dets(detections, annotations, num_classes, iou_thresh=0.5):
    from collections import defaultdict

    true_positives = []
    scores = []
    labels = []
    total_gt = 0

    for det, ann in zip(detections, annotations):
        detected = []
        gt_boxes = ann[:, 1:5] if len(ann) else np.zeros((0, 4))
        gt_cls = ann[:, 0] if len(ann) else np.zeros((0,))

        total_gt += len(gt_boxes)

        for pred in det:
            box = pred[:4]
            score = pred[4]
            cls = int(pred[5])

            scores.append(score)
            labels.append(cls)

            if len(gt_boxes):
                ious = compute_iou(box, gt_boxes)
                max_iou = ious.max()
                max_idx = ious.argmax()

                if max_iou > iou_thresh and max_idx not in detected and int(gt_cls[max_idx]) == cls:
                    true_positives.append(1)
                    detected.append(max_idx)
                else:
                    true_positives.append(0)
            else:
                true_positives.append(0)

    tp = np.array(true_positives)
    scores = np.array(scores)
    labels = np.array(labels)

    if len(tp) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "mAP": 0.0}

    indices = np.argsort(-scores)
    tp = tp[indices]
    labels = labels[indices]

    fp = 1 - tp
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)

    recall = tp_cumsum / (total_gt + 1e-6)
    precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    from .eval_metrics import voc_ap
    ap_dict = {}
    for c in range(num_classes):
        cls_mask = labels == c
        if cls_mask.any():
            cls_tp = tp[cls_mask]
            cls_fp = fp[cls_mask]
            cls_tp_cumsum = np.cumsum(cls_tp)
            cls_fp_cumsum = np.cumsum(cls_fp)

            rec = cls_tp_cumsum / (np.sum(cls_tp) + 1e-6)
            prec = cls_tp_cumsum / (cls_tp_cumsum + cls_fp_cumsum + 1e-6)
            ap_dict[c] = voc_ap(rec, prec)
        else:
            ap_dict[c] = 0.0

    return {
        "precision": float(precision[-1]),
        "recall": float(recall[-1]),
        "f1": float(f1[-1]),
        "mAP": float(np.mean(list(ap_dict.values()))),
        "per_class_ap": ap_dict,
    }

def compute_iou(box, boxes):
    inter_x1 = np.maximum(box[0], boxes[:, 0])
    inter_y1 = np.maximum(box[1], boxes[:, 1])
    inter_x2 = np.minimum(box[2], boxes[:, 2])
    inter_y2 = np.minimum(box[3], boxes[:, 3])

    inter_area = np.maximum(0, inter_x2 - inter_x1) * np.maximum(0, inter_y2 - inter_y1)
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    union = box_area + boxes_area - inter_area + 1e-6
    return inter_area / union
