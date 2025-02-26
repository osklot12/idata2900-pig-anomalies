from typing import Dict, Tuple, List, Any


class DarwinDecoder:
    """Decodes annotations stored in Darwin JSON format."""

    @staticmethod
    def get_annotations(json_data: Dict[str, Any]) -> Dict[int, List[Tuple[str, float, float, float, float]]]:
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

                h, w, x, y = DarwinDecoder._get_bounding_box_values(frame_data)

                if frame_index not in frame_annotations:
                    frame_annotations[frame_index] = []

                frame_annotations[frame_index].append(
                    (behavior, x, y, w, h)
                )

        return frame_annotations

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

    @staticmethod
    def get_frame_count(json_data: Dict[str, Any]) -> int:
        """
        Extracts the frame count from the Darwin JSON structure.

        Args:
            json_data (Dict[str, Any]): The annotation data loaded from a Darwin JSON file.

        Returns:
            int: The number of frames in the video. Returns -1 if not found.
        """
        result = -1

        slots = DarwinDecoder._get_slots(json_data)
        if slots and "frame_count" in slots[0]:
            result = slots[0]["frame_count"]

        return result

    @staticmethod
    def get_frame_dimensions(json_data: Dict[str, Any]) -> Tuple[int, int]:
        """
        Extracts the original image dimensions (width, height) from the Darwin JSON structure.

        Args:
            json_data (Dict[str, Any]): The annotation loaded from a Darwin JSON file.

        Returns:
            Tuple[int, int]: (width, height) of the original video frames. Defaults to (0, 0) if not found.
        """
        result = (0, 0)

        slots = DarwinDecoder._get_slots(json_data)
        if slots and "width" in slots[0] and "height" in slots[0]:
            result = slots[0]["width"], slots[0]["height"]
        return result

    @staticmethod
    def _get_slots(json_data):
        """Extracts the slots for the Darwin JSON structure."""
        return json_data.get("item", {}).get("slots", [])