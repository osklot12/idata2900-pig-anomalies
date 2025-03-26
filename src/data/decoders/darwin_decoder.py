from typing import Dict, Tuple, List, Optional

import json

from src.data.dataclasses.source_metadata import SourceMetadata
from src.data.label.label_parser import LabelParser
from src.data.dataclasses.annotated_bbox import AnnotatedBBox
from src.data.dataclasses.bbox import BBox
from src.data.dataclasses.frame_annotations import FrameAnnotations
from src.data.decoders.annotation_decoder import AnnotationDecoder
from src.typevars.enum_type import T_Enum


class DarwinDecoder(AnnotationDecoder):
    """Decodes annotations stored in Darwin JSON format."""

    def __init__(self, label_parser: LabelParser):
        """
        Initializes a DarwinDecoder instance.

        Args:
            label_parser (LabelParser): parses class labels
        """
        self._label_parser = label_parser

    def decode_annotations(self, raw_data: bytes) -> List[FrameAnnotations]:
        json_data = DarwinDecoder._get_json(raw_data)
        annotations = self._combine_annotations_by_frame(self._extract_annotations(json_data))
        return self._create_frame_annotation_list(
            annotations,
            self._create_source_metadata(json_data),
            DarwinDecoder.get_frame_count(raw_data)
        )

    @staticmethod
    def _get_json(raw_bytes: bytes):
        """Decodes raw bytes to a JSON format."""
        return json.loads(raw_bytes.decode("utf-8"))

    def _combine_annotations_by_frame(self, annotations: List[Dict]) -> Dict[int, List[AnnotatedBBox]]:
        """Groups annotations by their respective frame index."""
        frame_annotations: Dict[int, List[AnnotatedBBox]] = {}

        for annotation in annotations:
            for frame_index, bbox_annotations in self._parse_annotation_frames(annotation).items():
                if frame_index not in frame_annotations:
                    frame_annotations[frame_index] = []

                for bbox_annotation in bbox_annotations:
                    frame_annotations[frame_index].append(bbox_annotation)

        return frame_annotations

    def _parse_annotation_frames(self, annotation: Dict) -> Dict[int, List[AnnotatedBBox]]:
        """Parses and maps annotation data to frame indices."""
        frame_data_map = {}

        parsed_label = self._parse_label(self._extract_class_name(annotation))
        for frame_index, frame_data in self._extract_frame_data(annotation):
            frame_index = int(frame_index)
            bbox_annotation = self._create_bbox_annotation(parsed_label, frame_data)
            frame_data_map.setdefault(frame_index, []).append(bbox_annotation)

        return frame_data_map

    @staticmethod
    def _create_bbox_annotation(label: T_Enum, frame_data: Dict) -> AnnotatedBBox:
        """Creates a bounding box annotation for a given frame."""
        return AnnotatedBBox(
            cls=label,
            bbox=DarwinDecoder._create_bounding_box(frame_data)
        )

    def _parse_label(self, label: str) -> T_Enum:
        """Extracts and parses the class label from an annotation."""
        return self._label_parser.enum_from_str(label)

    @staticmethod
    def get_frame_count(raw_bytes: bytes) -> int:
        """
        Returns the total number of frames for the annotations file.

        Args:
            raw_bytes (bytes): the Darwin JSON data in raw bytes

        Returns:
            int: the total number of frames
        """
        result = -1

        slots = DarwinDecoder._extract_metadata_slots(DarwinDecoder._get_json(raw_bytes))
        if slots and "frame_count" in slots[0]:
            result = slots[0]["frame_count"]

        return result

    @staticmethod
    def get_frame_dimensions(raw_bytes: bytes) -> Tuple[int, int]:
        """
        Returns the frame dimensions (height, width) for the annotated frames.

        Args:
            raw_bytes (bytes): the Darwin JSON data in raw bytes

        Returns:
            Tuple[int, int]: the frame dimensions (height, width)
        """
        dimensions = DarwinDecoder._extract_frame_dimensions(DarwinDecoder._get_json(raw_bytes))
        if not dimensions:
            raise RuntimeError("Could not extract frame dimensions")

        return dimensions

    @staticmethod
    def _extract_frame_dimensions(json_data) -> Optional[Tuple[int, int]]:
        """Returns the frame dimensions from the metadata."""
        result = None

        slots = DarwinDecoder._extract_metadata_slots(json_data)
        if slots and "width" in slots[0] and "height" in slots[0]:
            result = slots[0]["width"], slots[0]["height"]

        return result

    @staticmethod
    def _create_source_metadata(json_data) -> SourceMetadata:
        """Creates and returns the source metadata."""
        return SourceMetadata(
            source_id=DarwinDecoder._extract_source_id(json_data),
            frame_resolution=DarwinDecoder._extract_frame_dimensions(json_data)
        )

    @staticmethod
    def _extract_source_id(json_data) -> str:
        """Extracts the source from the Darwin JSON format."""
        return json_data.get("item", {}).get("name", "unknown_source")

    @staticmethod
    def _extract_metadata_slots(json_data) -> List[Dict]:
        """Extracts the slots for the Darwin JSON structure."""
        return json_data.get("item", {}).get("slots", [])

    @staticmethod
    def _extract_annotations(json_data) -> List[Dict]:
        """Extracts the annotations for the Darwin JSON structure."""
        return json_data.get("annotations", [])

    @staticmethod
    def _extract_class_name(annotations: Dict) -> str:
        """Extracts the class for an annotation in Darwin JSON format."""
        return annotations.get("name", "unknown_class")

    @staticmethod
    def _extract_frame_data(annotation: Dict) -> List[Tuple[int, Dict]]:
        """Extracts the frames for an annotation in Darwin JSON format."""
        return list(annotation.get("frames", {}).items())

    @staticmethod
    def _create_bounding_box(frame_data) -> BBox:
        """Parses and retrieves bounding box values from a Darwin JSON annotation."""
        bbox = frame_data.get("bounding_box", {})
        return BBox(
            x=bbox.get("x", 0),
            y=bbox.get("y", 0),
            width=bbox.get("w", 0),
            height=bbox.get("h", 0)
        )

    @staticmethod
    def _create_frame_annotation_list(data: Dict[int, List[AnnotatedBBox]], source: SourceMetadata,
                                      frame_count: int) -> List[FrameAnnotations]:
        """Constructs a list of FrameAnnotation objects from grouped annotation data."""
        return [
            FrameAnnotations(
                source=source,
                index=frame_index,
                annotations=data.get(frame_index, []),
                end_of_stream=False
            )
            for frame_index in range(frame_count)
        ]
