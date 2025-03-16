from abc import ABC, abstractmethod

class AnnotationLoader(ABC):
    """An interface for annotation loaders."""

    @abstractmethod
    def load_annotation(self, annotation_id: str) -> dict:
        """
        Loads an annotation file.

        Args:
            annotation_id (str): the annotation identifier

        Returns:
            dict: the annotation file as a dictionary
        """
        raise NotImplementedError