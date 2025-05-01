import time

import torch
from torch.utils.tensorboard import SummaryWriter
from src.schemas.schemas.signed_schema import SignedSchema
from src.schemas.schemas.metric_schema import MetricSchema
from src.schemas.brokers.schema_broker import SchemaBroker
from src.schemas.training_metric_formater import TrainingMetricsFormatter
from src.models.training_metrics_calculator import TrainingMetricsCalculator

class YOLOXEvaluator:
    def __init__(self, model, dataset, broker: SchemaBroker, log_dir="runs/yolox_eval", issuer_id="yolox-evaluator"):
        self.model = model.eval()
        self.dataset = dataset
        self.broker = broker
        self.writer = SummaryWriter(log_dir=log_dir)
        self.issuer_id = issuer_id
        self.metric_calc = TrainingMetricsCalculator()

    def evaluate(self, num_batches=10):
        predictions_all = []
        targets_all = []

        with torch.no_grad():
            for i, batch in enumerate(self.dataset):
                if i >= num_batches:
                    break

                imgs = batch["img"].to(self.model.head.cls_preds[0].device)  # Send to correct device

                outputs = self.model(imgs)
                batch_size = len(imgs)

                for b in range(batch_size):
                    output = outputs[b]
                    boxes = output["boxes"] if "boxes" in output else output["bbox"]
                    predictions_all.append({
                        "boxes": boxes,
                        "scores": output["scores"],
                        "labels": output["cls"].int()
                    })

                    targets_all.append({
                        "boxes": batch["bboxes"][batch["batch_idx"] == b][:, :4],
                        "labels": batch["cls"][batch["batch_idx"] == b]
                    })

        metrics = self.metric_calc.calculate(predictions_all, targets_all)

        self.writer.add_scalar("eval/mAP50", metrics["mAP@0.5"], 0)
        self.writer.add_scalar("eval/recall", metrics["Recall@100"], 0)
        self.writer.add_scalar("eval/F1", metrics["F1"], 0)
        self.writer.flush()
        self.writer.close()

        schema = SignedSchema(
            signature=self.issuer_id,
            schema=MetricSchema(metrics=metrics, timestamp=time.time())
        )
        self.broker.notify(schema)

        return metrics
