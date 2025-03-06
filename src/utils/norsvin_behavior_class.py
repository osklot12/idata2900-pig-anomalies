from enum import Enum, auto

from src.data.annotation_enum_parser import AnnotationEnumParser


class NorsvinBehaviorClass(Enum, AnnotationEnumParser):
    """An enumeration of Norsvin's behavioral classes."""

    TAIL_BITING = auto()
    """Behavior where pig bites tail."""

    EAR_BITING = auto()
    """Behavior where pig bites other pig's ear."""

    BELLY_NOSING = auto()
    """Behavior where pig rubs their nose on other pigs' bellies."""

    TAIL_DOWN = auto()
    """Behavior where pigs puts their tail down."""

    @staticmethod
    def enum_from_str(self, label: str):
        mapping = {
            "g2b_tailbiting": NorsvinBehaviorClass.TAIL_BITING,
            "g2b_earbiting": NorsvinBehaviorClass.EAR_BITING,
            "g2b_bellynosing": NorsvinBehaviorClass.BELLY_NOSING,
            "g2b_taildown": NorsvinBehaviorClass.TAIL_DOWN,
        }
        return mapping.get(label, None)