from pathlib import Path

# define project root dynamically
PROJECT_ROOT = Path(__file__).resolve().parent.parent

def get_test_path(relative_path: str) -> Path:
    """Returns an absolute path for test files, regardless of working directory."""
    return PROJECT_ROOT / relative_path