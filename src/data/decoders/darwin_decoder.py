from typing import Dict, Tuple, List, Any

from src.data.decoders.bbox_decoder import BBoxDecoder


class DarwinDecoder(BBoxDecoder):
    """Decodes annotations stored in Darwin JSON format."""

    def get_annotations(self) -> Dict[int, List[Tuple[str, float, float, float, float]]]:
        frame_annotations = {}

        for annotation in self.json_data.get("annotations", []):
            behavior = annotation.get("name", "unknown_behavior")

            for frame_index, frame_data in annotation.get("frames", {}).items():
                frame_index = int(frame_index)

                h, w, x, y = DarwinDecoder._get_bounding_box_values(frame_data)

                if frame_index not in frame_annotations:
                    frame_annotations[frame_index] = []

                frame_annotations[frame_index].append(
                    (behavior, x, y, w, h)
                )

        return frame_annotations

    def get_frame_count(self) -> int:
        result = -1

        slots = DarwinDecoder._get_slots(self.json_data)
        if slots and "frame_count" in slots[0]:
            result = slots[0]["frame_count"]

        return result

    def get_frame_dimensions(self) -> Tuple[int, int]:
        result = (0, 0)

        slots = DarwinDecoder._get_slots(self.json_data)
        if slots and "width" in slots[0] and "height" in slots[0]:
            result = slots[0]["width"], slots[0]["height"]
        return result

    @staticmethod
    def _get_slots(json_data):
        """Extracts the slots for the Darwin JSON structure."""
        return json_data.get("item", {}).get("slots", [])

    @staticmethod
    def _get_bounding_box_values(frame_data):
        """Parses and retrieves bounding box values from a Darwin JSON annotation."""
        bbox = frame_data.get("bounding_box", {})
        x, y, w, h = (
            bbox.get("x", 0),
            bbox.get("y", 0),
            bbox.get("w", 0),
            bbox.get("h", 0)
        )
        return h, w, x, y