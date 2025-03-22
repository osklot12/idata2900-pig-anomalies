from abc import ABC, abstractmethod

from src.data.label.label_parser import LabelParser


class LabelParserFactory(ABC):
    """An interface for label parser factories."""

    @abstractmethod
    def create_label_parser(self) -> LabelParser:
        """
        Creates a LabelParser instance.

        Returns:
            LabelParser: the label parser instance
        """
        raise NotImplementedError