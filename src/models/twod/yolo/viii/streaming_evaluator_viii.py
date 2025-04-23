
import torch
import numpy as np
from tqdm import tqdm

from ultralytics.utils.metrics import ap_per_class
from ultralytics.utils.ops import xywh2xyxy


class StreamingEvaluatorVIII:
    def __init__(self, model, dataloader, device, num_classes, iou_thresh=0.5):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.num_classes = num_classes
        self.iou_thresh = iou_thresh

    def evaluate(self):
        self.model.eval()
        seen = 0
        stats = []

        pbar = tqdm(self.dataloader, desc="Evaluating", leave=False)
        for batch in pbar:
            imgs = batch["img"].to(self.device, non_blocking=True).float()
            cls_targets = batch["cls"]
            bboxes = batch["bboxes"]
            batch_idx = batch["batch_idx"]

            with torch.no_grad():
                preds = self.model(imgs)[0]

            for i in range(len(imgs)):
                tcls = cls_targets[batch_idx == i]
                tbox = bboxes[batch_idx == i]

                if len(tbox):
                    tbox = xywh2xyxy(tbox) * imgs.shape[2]

                pred = preds[preds[:, 0] == i][:, 1:] if len(preds) else torch.zeros((0, 6), device=self.device)

                if len(pred):
                    predn = pred.clone()
                    predn[:, :4] = predn[:, :4].clamp(0, imgs.shape[2])
                else:
                    predn = torch.zeros((0, 6), device=self.device)

                correct = self._match_predictions(predn, tbox, tcls)
                stats.append((correct.cpu(), predn[:, 4].cpu(), predn[:, 5].cpu(), tcls.cpu()))
                seen += 1

        stats = [np.concatenate(x, 0) for x in zip(*stats)] if stats else ([], [], [], [])
        metrics = {}

        if len(stats) and stats[0].any():
            p, r, ap, f1, ap_class = ap_per_class(*stats, plot=False, save_dir=None)
            metrics = {
                "precision": round(p.mean().item(), 4),
                "recall": round(r.mean().item(), 4),
                "mAP50": round(ap[:, 0].mean().item(), 4),
                "mAP50-95": round(ap.mean().item(), 4),
            }
        else:
            metrics = {
                "precision": 0.0,
                "recall": 0.0,
                "mAP50": 0.0,
                "mAP50-95": 0.0,
            }

        return metrics

    def _match_predictions(self, preds, tbox, tcls):
        correct = torch.zeros(len(preds), dtype=torch.bool, device=preds.device)
        if not len(tbox):
            return correct

        detected = []
        for i, p in enumerate(preds):
            iou = self._box_iou(p[None, :4], tbox)
            iou_max, iou_idx = iou.max(1)

            if iou_max > self.iou_thresh and iou_idx.item() not in detected and p[5] == tcls[iou_idx]:
                correct[i] = True
                detected.append(iou_idx.item())

        return correct

    def _box_iou(self, boxes1, boxes2):
        def box_area(box):
            return (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])

        area1 = box_area(boxes1)
        area2 = box_area(boxes2)

        inter = (torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
                 - torch.max(boxes1[:, None, :2], boxes2[:, :2])).clamp(0).prod(2)
        return inter / (area1[:, None] + area2 - inter + 1e-6)
