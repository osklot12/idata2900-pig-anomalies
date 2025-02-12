import os
import json
import requests
from google.auth.transport.requests import Request
from google.oauth2 import service_account
from functools import lru_cache
from io import BytesIO


class GCPDataLoader:
    """
    Handles authentication and efficient data loading from Google Cloud Storage (GCS).
    """

    def __init__(self, bucket_name: str, credentials_path: str):
        """
        Initializes the GCPDataLoader.

        :param bucket_name: The name of the GCS bucket.
        :param credentials_path: The path to the service account JSON credentials file.
        """
        self.bucket_name = bucket_name
        self.credentials_path = credentials_path
        self.token = None
        self.creds = None
        self.authenticate()

    def authenticate(self):
        """Authenticates with GCS using a service account."""
        if not os.path.exists(self.credentials_path):
            raise FileNotFoundError(f"Service account file not found: {self.credentials_path}")

        self.creds = service_account.Credentials.from_service_account_file(
            self.credentials_path, scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        self.refresh_token()

    def refresh_token(self):
        """Refreshes the token if needed."""
        if not self.creds or not self.creds.valid:
            self.creds.refresh(Request())
        self.token = self.creds.token

    def get_access_token(self):
        """Returns a valid access token, refreshing if necessary."""
        self.refresh_token()
        return self.token

    @lru_cache(maxsize=1)
    def fetch_all_files(self):
        """Retrieves all file names in the bucket (caches results to reduce API calls)."""
        headers = {"Authorization": f"Bearer {self.get_access_token()}"}
        url = f"https://www.googleapis.com/storage/v1/b/{self.bucket_name}/o"

        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            raise RuntimeError(f"Failed to fetch files: {response.text}")

        return [file["name"] for file in response.json().get("items", [])]

    def list_files(self, prefix="", file_extension=""):
        """Lists files in the bucket with optional prefix and extension filtering."""
        all_files = self.fetch_all_files()
        return [file for file in all_files if file.startswith(prefix) and file.endswith(file_extension)]

    def stream_video(self, blob_name: str, chunk_size: int = 1920 * 1080) -> BytesIO:
        """
        Streams a video file from GCS in chunks to avoid loading large files into memory.

        :param blob_name: The name of the video file in GCS.
        :param chunk_size: The size of each chunk in bytes.
        :return: BytesIO stream of the file.
        """
        headers = {"Authorization": f"Bearer {self.get_access_token()}"}
        file_url = f"https://storage.googleapis.com/{self.bucket_name}/{blob_name}"

        response = requests.get(file_url, headers=headers, stream=True, timeout=10)
        if response.status_code != 200:
            raise RuntimeError(f"Failed to stream {blob_name}. Error: {response.status_code} | {response.text}")

        return BytesIO(b"".join(response.iter_content(chunk_size)))

    def download_json(self, blob_name: str) -> dict:
        """
        Downloads a JSON file from GCS and returns the parsed content.

        :param blob_name: The name of the JSON file in GCS.
        :return: The parsed JSON content as a dictionary.
        """
        headers = {"Authorization": f"Bearer {self.get_access_token()}"}
        file_url = f"https://storage.googleapis.com/{self.bucket_name}/{blob_name}"

        response = requests.get(file_url, headers=headers)
        if response.status_code != 200:
            raise RuntimeError(f"Failed to download {blob_name}. Error: {response.status_code} | {response.text}")

        return response.json()