from typing import List

from src.data.dataset.dataset_source import DatasetSource
from src.data.gcs_bucket_client import GCSBucketClient


class GCSDatasetSource(GCSBucketClient, DatasetSource):
    """Handles file listing in a Google Cloud Storage bucket."""

    def list_files(self) -> List[str]:
        response = self._make_request(
            f"https://www.googleapis.com/storage/v1/b/{self._bucket_name}/o"
        )

        return [file["name"] for file in response.json().get("items", [])]