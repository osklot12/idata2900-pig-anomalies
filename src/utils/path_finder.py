from pathlib import Path

# define project root dynamically
PROJECT_ROOT = Path(__file__).resolve().parents[2]

class PathFinder:
    """Utility class for finding and managing file paths, ensuring consistency across the project."""

    @staticmethod
    def get_abs_path(relative_path: str) -> Path:
        """
        Returns an absolute path for files, regardless of working directory.

        Args:
            :param relative_path: The path relative to the project root directory.

        Returns:
            Path: The absolute path.
        """
        return PROJECT_ROOT / relative_path