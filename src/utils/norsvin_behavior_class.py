from enum import Enum, auto


class NorsvinBehaviorClass(Enum):
    """An enumeration of Norsvin's behavioral classes."""

    TAIL_BITING = 0
    """Behavior where pig bites tail."""

    EAR_BITING = 1
    """Behavior where pig bites other pig's ear."""

    BELLY_NOSING = 2
    """Behavior where pig rubs their nose on other pigs' bellies."""

    TAIL_DOWN = 3
    """Behavior where pigs puts their tail down."""

    _LABEL_MAP = {
        "g2b_tailbiting": TAIL_BITING,
        "g2b_earbiting": EAR_BITING,
        "g2b_bellynosing": BELLY_NOSING,
        "g2b_taildown": TAIL_DOWN,
    }

    @classmethod
    def get_label_map(cls):
        return {
            "g2b_tailbiting": NorsvinBehaviorClass.TAIL_BITING,
            "g2b_earbiting": NorsvinBehaviorClass.EAR_BITING,
            "g2b_bellynosing": NorsvinBehaviorClass.BELLY_NOSING,
            "g2b_taildown": NorsvinBehaviorClass.TAIL_DOWN,
        }
