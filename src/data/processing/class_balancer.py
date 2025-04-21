import copy
from typing import Dict, List
import math
import random

from src.data.dataclasses.annotated_frame import AnnotatedFrame
from src.data.processing.processor import Processor
from src.data.typevars.enum_type import T_Enum


class ClassBalancer(Processor[AnnotatedFrame, List[AnnotatedFrame]]):
    """Pipeline component for balancing classes."""

    def __init__(self, class_counts: Dict[T_Enum, int], max_samples_per: int):
        """
        Initializes a ClassBalancer instance.

        Args:
            class_counts (Dict[T_Enum, int]): a dictionary containing the count of each class
            max_samples_per (int): the maximum number of copies allowed per instance
        """
        self._class_counts = class_counts
        self._dominant_class: T_Enum = self._get_dominant_class()
        self._factors = self._compute_factors()
        self._max_samples_per = max_samples_per

    def process(self, data: AnnotatedFrame) -> List[AnnotatedFrame]:
        frames = [data]

        outputs = 1
        if data is not None and len(data.annotations) > 0:
            classes = set()
            for annotation in data.annotations:
                classes.add(annotation.cls)

            if not self._dominant_class in classes:
                factors = {cls: self._factors[cls] for cls in classes if cls in self._factors}
                factor = min(factors.values())

                base = math.floor(factor)
                add = 1 if random.random() < (factor - base) else 0
                outputs = min(base + add, self._max_samples_per)

        for _ in range(outputs - 1):
            frames.append(copy.deepcopy(data))

        return frames

    def _get_dominant_class(self) -> T_Enum:
        """Finds the dominant class."""
        dominant_class = None
        dominant_count = 0
        for k, v in self._class_counts.items():
            if v > dominant_count:
                dominant_class = k
                dominant_count = v

        return dominant_class

    def _compute_factors(self) -> Dict[T_Enum, float]:
        """Computes the multiplication factor for each class."""
        factors = {}
        for k, v in self._class_counts.items():
            factors[k] = self._class_counts[self._dominant_class] / v

        return factors