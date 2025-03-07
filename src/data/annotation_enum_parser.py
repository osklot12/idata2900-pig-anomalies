from abc import ABC, abstractmethod

class AnnotationEnumParser(ABC):
    """Interface for converting annotation labels to enums."""

    @staticmethod
    @abstractmethod
    def enum_from_str(label: str):
        """Converts an annotation label string to an enum."""
        raise NotImplementedError