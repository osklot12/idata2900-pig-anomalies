from typing import Dict

from src.data.label.label_parser import LabelParser
from src.typevars.enum_type import T_Enum


class SimpleLabelParser(LabelParser):
    """A simple annotation parser."""

    def __init__(self, parse_map: Dict[str, T_Enum]):
        """
        Initializes a SimpleLabelParser instance.

        Args:
            parse_map (Dict[str, T_Enum]): the map used for parsing
        """
        self._parse_map = parse_map

    def enum_from_str(self, label: str) -> T_Enum:
        return self._parse_map.get(label, None)