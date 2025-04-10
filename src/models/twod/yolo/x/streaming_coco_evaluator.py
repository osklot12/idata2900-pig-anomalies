from typing import List, Dict, Tuple

from yolox.evaluators import COCOEvaluator
from yolox.layers import COCOeval_opt


class StreamingCOCOEvaluator(COCOEvaluator):
    """COCO Evaluator for streaming data with safe fallback for batch_size=None."""

    def evaluate_prediction(self, data_list: List, statistics: Tuple[float, float, int]) -> Dict[str, float]:
        """
        Patched version of evaluate_prediction to safely handle None batch size from streaming DataLoaders.
        This method keeps the default behavior for computing coco_gt and coco_dt.
        """
        # First: call the original implementation to ensure self.coco_gt and self.coco_dt are set
        super().evaluate_prediction(data_list, statistics)

        infer_time, nms_time, n_samples = statistics
        batch_size = self.dataloader.batch_size or 1
        a_infer_time = 1000 * infer_time / (n_samples * batch_size)
        a_nms_time = 1000 * nms_time / (n_samples * batch_size)

        # Now evaluate with COCO
        coco_eval = COCOeval_opt(self.coco_gt, self.coco_dt, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        self.map_50_95 = coco_eval.stats[0]
        self.map_50 = coco_eval.stats[1]

        return {
            "infer_time(ms/img)": round(a_infer_time, 2),
            "nms_time(ms/img)": round(a_nms_time, 2),
            "mAP@0.5:0.95": round(self.map_50_95, 4),
            "mAP@0.5": round(self.map_50, 4),
        }