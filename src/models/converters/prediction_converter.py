from typing import List, Dict
from src.models.converters.converter_interfaces import PredictionsToListConverter

class PredictionConverter(PredictionsToListConverter):
    def convert_preds_to_list(self, preds) -> List[Dict]:
        out = []
        for pred in preds:
            boxes = pred.boxes
            out.append({
                "boxes": boxes.xyxy.cpu().tolist(),
                "scores": boxes.conf.cpu().tolist(),
                "labels": boxes.cls.cpu().tolist(),
            })
        return out
