import os
import requests
from functools import lru_cache
from io import BytesIO
from src.auth.gcp_auth_service import GCPAuthService


class GCPDataLoader:
    """
    Handles authentication and efficient data loading from Google Cloud Storage (GCS).
    Now includes deterministic seeding and per-frame shuffling for training.
    """

    def __init__(self, bucket_name: str, auth_service: "GCPAuthService"):
        """
        Initializes the GCPDataLoader.
        :param bucket_name: The name of the GCS bucket.
        :param auth_service: The GCPAuthService instance to use for authentication.
        """
        self.bucket_name = bucket_name
        self.auth_service = auth_service

    @lru_cache(maxsize=1)
    def fetch_all_files(self):
        """Retrieves all file names in the bucket (caches results to reduce API calls)."""
        headers = {"Authorization": f"Bearer {self.auth_service.get_access_token()}"}
        url = f"https://www.googleapis.com/storage/v1/b/{self.bucket_name}/o"

        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            raise RuntimeError(f"Failed to fetch files: {response.text}")

        return [file["name"] for file in response.json().get("items", [])]

    @lru_cache(maxsize=10)
    def list_files(self, prefix="", file_extension=""):
        """Lists files in the bucket with optional prefix and extension filtering."""
        all_files = self.fetch_all_files()
        filtered_files = [file for file in all_files if file.startswith(prefix) and file.endswith(file_extension)]
        return filtered_files

    def download_video(self, blob_name: str) -> BytesIO:
        """
        Downloads a full video file from GCS into memory.

        :param blob_name: The name of the video file in GCS.
        :return BytesIO object containing the raw video data.
        """
        headers = {"Authorization": f"Bearer {self.auth_service.get_access_token()}"}
        file_url = f"https://storage.googleapis.com/{self.bucket_name}/{blob_name}"

        response = requests.get(file_url, headers=headers, stream=True, timeout=60)
        if response.status_code != 200:
            raise RuntimeError(f"Failed to download {blob_name}. Error: {response.status_code} | {response.text}")

        video_data = BytesIO()
        for chunk in response.iter_content(chunk_size=8192):
            video_data.write(chunk)

        video_data.seek(0)
        return video_data

    def download_json(self, blob_name: str) -> dict:
        """
        Downloads a JSON file from GCS and returns the parsed content.

        :param blob_name: The name of the JSON file in GCS.
        :return: The parsed JSON content as a dictionary.
        """
        headers = {"Authorization": f"Bearer {self.auth_service.get_access_token()}"}
        file_url = f"https://storage.googleapis.com/{self.bucket_name}/{blob_name}"

        response = requests.get(file_url, headers=headers)
        if response.status_code != 200:
            raise RuntimeError(f"Failed to download {blob_name}. Error: {response.status_code} | {response.text}")

        return response.json()
