from src.data.annotation_enum_parser import AnnotationEnumParser
from src.utils.norsvin_behavior_class import NorsvinBehaviorClass


class NorsvinAnnotationParser(AnnotationEnumParser):
    """Parses Norsvin behavior classes from string to enums."""

    @staticmethod
    def enum_from_str(label: str):
        mapping = {
            "g2b_tailbiting": NorsvinBehaviorClass.TAIL_BITING,
            "g2b_earbiting": NorsvinBehaviorClass.EAR_BITING,
            "g2b_bellynosing": NorsvinBehaviorClass.BELLY_NOSING,
            "g2b_taildown": NorsvinBehaviorClass.TAIL_DOWN,
        }
        return mapping.get(label, None)