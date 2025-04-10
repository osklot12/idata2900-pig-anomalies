from abc import ABC, abstractmethod

class Identifier(ABC):
    """Interface for instance identifiers."""

    @abstractmethod
    def identify(self, video: str, annotations: str) -> str:
        """
        Returns an ID for the video - annotations pair.

        Args:
            video (str): the video ID
            annotations (str): the annotations ID

        Returns:
            str: an ID for the video - annotations pair
        """
        raise NotImplementedError