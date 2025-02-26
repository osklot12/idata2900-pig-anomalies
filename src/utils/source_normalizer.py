import os

class SourceNormalizer:
    """Utility class to standardize source identifiers across the dataset pipeline."""

    @staticmethod
    def normalize(source: str) -> str:
        """
        Extracts the base filename from a path and removes the extension.

        :param source: Full source path.
        :return: Normalized source ID.
        """
        base = os.path.basename(source)
        name, _ = os.path.splitext(base)
        return name