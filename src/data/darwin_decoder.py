from typing import Dict, Tuple, List


class DarwinDecoder:
    """Decodes annotations stored in Darwin JSON format."""

    @staticmethod
    def decode(json_data) -> Dict[int, List[Tuple[str, float, float, float, float]]]:
        """
        Decodes Darwin JSON annotations into a dictionary where:
        - Key: Frame index (int)
        - Value: List of tuples (behavior, x, y, w, h)

        Args:
            json_data (Dict[str, Any]): The annotation data loaded from a Darwin JSON file.

        Returns:
            Dict[int, List[Tuple[str, float, float, float, float]]]: The decoded annotation data.
        """
        frame_annotations = {}

        for annotation in json_data.get("annotations", []):
            behavior = annotation.get("name", "unknown_behavior")

            for frame_index, frame_data in annotation.get("frames", {}).items():
                frame_index = int(frame_index)

                bbox = frame_data.get("bounding_box", {})
                x, y, w, h = (
                    bbox.get("x", 0),
                    bbox.get("y", 0),
                    bbox.get("w", 0),
                    bbox.get("h", 0)
                )

                if frame_index not in frame_annotations:
                    frame_annotations[frame_index] = []

                frame_annotations[frame_index].append(
                    (behavior, x, y, w, h)
                )

        return frame_annotations