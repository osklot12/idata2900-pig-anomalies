from src.data.dataset.identifiers.identifier import Identifier
from src.data.parsing.file_base_name_parser import FileBaseNameParser


class BaseNameIdentifier(Identifier):
    """Identifier that creates basename IDs."""

    def __init__(self):
        """Initializes a BaseNameIdentifier instance."""
        self._parser = FileBaseNameParser()

    def identify(self, video: str, annotations: str) -> str:
        video_base_name = self._parser.parse_string(video)
        annotations_base_name = self._parser.parse_string(annotations)
        if not video_base_name == annotations_base_name:
            raise ValueError(f"video ({video}) and annotations ({annotations}) base names do not match")

        return video_base_name