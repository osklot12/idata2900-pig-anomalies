from enum import Enum, auto

class NorsvinBehaviorClass(Enum):
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
    def from_json_label(label: str) -> "NorsvinBehaviorClass":
        """Converts JSON behavior labels to enum values."""
        mapping = {
            "g2b_tailbiting": NorsvinBehaviorClass.TAIL_BITING,
            "g2b_earbiting": NorsvinBehaviorClass.EAR_BITING,
            "g2b_bellynosing": NorsvinBehaviorClass.BELLY_NOSING,
            "g2b_taildown": NorsvinBehaviorClass.TAIL_DOWN,
        }
        return mapping.get(label, None)