import torch
from src.models.training_metrics_calculator import TrainingMetricsCalculator

class YOLOEvaluator:
    def __init__(self, model, dataloader):
        self.model = model
        self.dataloader = dataloader
        self.metric_calculator = TrainingMetricsCalculator()
        self.metrics = {}

    def evaluate(self):
        self.model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.dataloader):
                imgs = batch["img"]

                results = self.model(imgs)

                if isinstance(results, (tuple, list)):
                    predictions = results[0]
                elif hasattr(results, "pred"):
                    predictions = results.pred
                else:
                    raise ValueError(f"Unsupported output type from model: {type(results)}")

                print(f"\n🔍 Batch {batch_idx} → Predictions: {len(predictions)} samples")

                for i, pred in enumerate(predictions):
                    if pred.numel() > 0:
                        pred_boxes = pred[:, :4]
                        pred_scores = pred[:, 4]
                        pred_labels = pred[:, 5].long()
                    else:
                        pred_boxes = torch.empty((0, 4))
                        pred_scores = torch.empty((0,))
                        pred_labels = torch.empty((0,), dtype=torch.int64)

                    # 🧠 Debug: Prediction shapes
                    print(
                        f"  🟦 Pred[{i}] boxes: {pred_boxes.shape}, scores: {pred_scores.shape}, labels: {pred_labels.shape}")

                    all_preds.append({
                        "boxes": pred_boxes,
                        "scores": pred_scores,
                        "labels": pred_labels,
                    })

                    # 🧠 Debug: Target matching
                    matched_box_indices = batch["batch_idx"] == i
                    matched_boxes = batch["bboxes"][matched_box_indices][:, :4]
                    matched_labels = batch["cls"][matched_box_indices].long()

                    print(f"  🎯 Target[{i}] boxes: {matched_boxes.shape}, labels: {matched_labels.shape}")

                    all_targets.append({
                        "boxes": matched_boxes,
                        "labels": matched_labels,
                    })

        print(f"\n✅ Total predictions: {len(all_preds)} | Total targets: {len(all_targets)}")
        metrics = self.metric_calculator.calculate(all_preds, all_targets)
        print("📊 Evaluation Metrics:", metrics)

        self.metrics = metrics  # 🔧 Save result to object
        return metrics
