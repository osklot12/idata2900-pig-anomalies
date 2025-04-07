from src.data.gcs_bucket_client import GCSBucketClient
from src.data.loading.loaders.video_file_loader import VideoFileLoader


class GCSVideoLoader(GCSBucketClient, VideoFileLoader):
    """Handles downloading video files from Google Cloud Storage."""

    def load_video_file(self, video_id: str) -> bytes:
        video_data = self._make_request(
            self._get_file_url(video_id)
        ).content
        print(f"[GCSVideoLoader] Loaded video file {video_id}")
        return bytearray(video_data)