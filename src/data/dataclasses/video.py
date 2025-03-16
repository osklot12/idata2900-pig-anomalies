from dataclasses import dataclass

from src.data.loading.loaders.video_loader import VideoLoader


@dataclass(frozen=True)
class Video:
    """Holds video-related information in an immutable structure."""
    id: str
    loader: VideoLoader