from src.data.gcs_bucket_client import GCSBucketClient
from src.data.loading.loaders.video_file_data_loader import VideoFileDataLoader


class GCSVideoLoader(GCSBucketClient, VideoFileDataLoader):
    """Handles downloading video files from Google Cloud Storage."""

    def load_video_file_data(self, video_id: str) -> bytes:
        video_data = self._make_request(
            self._get_file_url(video_id)
        ).content

        return bytearray(video_data)