from abc import ABC, abstractmethod

from src.typevars.enum_type import T_Enum


class LabelParser(ABC):
    """Interface for converting annotation labels to enums."""

    @abstractmethod
    def enum_from_str(self, label: str) -> T_Enum:
        """
        Converts a string label to an enum.

        Args:
            label (str): the label to convert

        Returns:
            T_Enum: the converted enum, or None if label is not recognized
        """
        raise NotImplementedError