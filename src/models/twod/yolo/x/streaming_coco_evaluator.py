from typing import List, Dict, Tuple

from yolox.evaluators import COCOEvaluator
from yolox.layers import COCOeval_opt


class StreamingCOCOEvaluator(COCOEvaluator):
    """COCO Evaluator for streaming data with safe fallback for batch_size=None."""

    def evaluate(self, model, distributed=False, half=False, trt_file=None, decoder=None, test_size=None,
                 return_outputs=False):
        """
        Override the full evaluate() method to safely use our patched prediction logic.
        """
        # This calls the original implementation, including the setup of self.coco_gt and self.coco_dt.
        outputs = super().evaluate(model, distributed, half, trt_file, decoder, test_size, return_outputs)
        return outputs

    def evaluate_prediction(self, data_list: List, statistics: Tuple[float, float, int]) -> Dict[str, float]:
        infer_time, nms_time, n_samples = statistics

        a_infer_time, a_nms_time = self._compute_average_times(infer_time, nms_time, n_samples)
        self._run_coco_evaluation()

        return self._build_summary(a_infer_time, a_nms_time)

    def _compute_average_times(self, infer_time: float, nms_time: float, n_samples: int) -> Tuple[float, float]:
        batch_size = self.dataloader.batch_size or 1
        a_infer_time = 1000 * infer_time / (n_samples * batch_size)
        a_nms_time = 1000 * nms_time / (n_samples * batch_size)
        return a_infer_time, a_nms_time

    def _run_coco_evaluation(self):
        coco_eval = COCOeval_opt(self.coco_gt, self.coco_dt, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        self.map_50_95 = coco_eval.stats[0]
        self.map_50 = coco_eval.stats[1]

    def _build_summary(self, a_infer_time: float, a_nms_time: float) -> Dict[str, float]:
        return {
            "infer_time(ms/img)": round(a_infer_time, 2),
            "nms_time(ms/img)": round(a_nms_time, 2),
            "mAP@0.5:0.95": round(self.map_50_95, 4),
            "mAP@0.5": round(self.map_50, 4),
        }