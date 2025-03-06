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