import torch

def postprocess(prediction, num_classes, conf_thres=0.001, nms_thres=0.65):
    """
    Runs Non-Maximum Suppression (NMS) on inference results.

    Args:
        prediction (Tensor): Raw predictions from model [batch, num_boxes, 5 + num_classes]
        num_classes (int): Number of classes
        conf_thres (float): Confidence threshold
        nms_thres (float): IoU threshold for NMS

    Returns:
        List of detections, one per image: (x1, y1, x2, y2, conf, cls)
    """
    output = [None for _ in range(len(prediction))]
    for i, image_pred in enumerate(prediction):
        if not image_pred.size(0):
            continue

        # Compute conf = obj_conf * cls_conf
        image_pred[:, 5:] *= image_pred[:, 4:5]

        # Get class with max confidence
        conf, cls = image_pred[:, 5:].max(1, keepdim=True)
        detections = torch.cat((image_pred[:, :4], conf, cls.float()), 1)

        # Filter by conf threshold
        mask = conf.view(-1) > conf_thres
        detections = detections[mask]
        if not detections.size(0):
            continue

        # Perform NMS
        boxes = detections[:, :4]
        scores = detections[:, 4]
        classes = detections[:, 5]

        keep = torch.ops.torchvision.nms(boxes, scores, nms_thres)
        output[i] = detections[keep]

    return output
