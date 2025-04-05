from typing import List

from src.data.dataset.registries.file_registry import FileRegistry
from src.data.gcs_bucket_client import GCSBucketClient


class GCSFileRegistry(GCSBucketClient, FileRegistry):
    """Handles file listing in a Google Cloud Storage bucket."""

    def get_file_paths(self) -> set[str]:
        response = self._make_request(
            f"https://www.googleapis.com/storage/v1/b/{self._bucket_name}/o"
        )

        return {file["name"] for file in response.json().get("items", [])}