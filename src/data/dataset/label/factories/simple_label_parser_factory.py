from typing import Dict

from src.data.dataset.label.factories.label_parser_factory import LabelParserFactory
from src.data.dataset.label.label_parser import LabelParser
from src.data.dataset.label.simple_label_parser import SimpleLabelParser
from src.data.typevars.enum_type import T_Enum


class SimpleLabelParserFactory(LabelParserFactory):
    """A factory for creating SimpleLabelParser instances."""

    def __init__(self, label_map: Dict[str, T_Enum]):
        """
        Initializes a SimpleLabelParserFactory instance.

        Args:
            label_map (Dict[str, T_Enum]): a mapping of strings to label enums
        """
        self._label_map = label_map

    def create_label_parser(self) -> LabelParser:
        return SimpleLabelParser(self._label_map.copy())