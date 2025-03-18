from abc import ABC, abstractmethod

class AnnotationJson(ABC):
    """An interface for annotations."""

    @abstractmethod
    def get_id(self) -> str:
        """
        Returns the annotation ID.

        Returns:
            str: the annotation ID
        """
        raise NotImplementedError

    @abstractmethod
    def get_data(self) -> dict:
        """
        Returns the annotation data.

        Returns:
            dict: the annotation data
        """
        raise NotImplementedError