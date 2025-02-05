import os
import json
import requests
import tensorflow as tf
from google.auth.transport.requests import Request
from google.oauth2 import service_account
from io import BytesIO

# 🔹 CONFIGURE YOUR BUCKET
BUCKET_NAME = "norsvin-g2b-behavior-prediction"
ANNOTATIONS_PATH = "g2b_behaviour/releases/g2b-prediction/annotations"
SEED = 42  # Ensures deterministic shuffling
CREDENTIALS_FILE = r"C:\Users\chris\Documents\Skole2025\IDATA2900\idata2900-pig-anomalies\.secrets\norsvin-research-3706-gn2bhv-06eecdfeb6aa.json"
SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]


class GCPStorageHandler:
    """Handles authentication and interactions with Google Cloud Storage."""

    def __init__(self, bucket_name=BUCKET_NAME, credentials_path=CREDENTIALS_FILE):
        self.bucket_name = bucket_name
        self.credentials_path = credentials_path
        self.token = None

    def authenticate(self):
        """Authenticate using the service account and refresh the token."""
        if not os.path.exists(self.credentials_path):
            raise FileNotFoundError(f"❌ Service account file not found: {self.credentials_path}")

        creds = service_account.Credentials.from_service_account_file(
            self.credentials_path, scopes=SCOPES
        )

        creds.refresh(Request())  # Ensure token is fresh
        self.token = creds.token

    def get_access_token(self):
        """Returns a valid access token (refreshes if needed)."""
        if not self.token:
            self.authenticate()
        return self.token

    def list_files(self, prefix="", file_extension=""):
        """List all files in the GCS bucket, optionally filtering by prefix and extension."""
        self.authenticate()
        headers = {"Authorization": f"Bearer {self.get_access_token()}"}
        url = f"https://www.googleapis.com/storage/v1/b/{self.bucket_name}/o?prefix={prefix}"

        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            print(f"❌ Error listing files: {response.text}")
            return []

        files = [file["name"] for file in response.json().get("items", []) if file["name"].endswith(file_extension)]
        return files

    def stream_video(self, blob_name):
        """Stream a video file from GCS into memory."""
        self.authenticate()
        headers = {"Authorization": f"Bearer {self.get_access_token()}"}
        file_url = f"https://storage.googleapis.com/{self.bucket_name}/{blob_name}"

        response = requests.get(file_url, headers=headers, stream=True, timeout=10)

        if response.status_code != 200:
            print(f"❌ Failed to stream {blob_name}. Error: {response.status_code} | {response.text}")
            return None

        return BytesIO(response.content)

    def download_json(self, blob_name):
        """Download a JSON file from GCS and return the parsed content."""
        self.authenticate()
        headers = {"Authorization": f"Bearer {self.get_access_token()}"}
        file_url = f"https://storage.googleapis.com/{self.bucket_name}/{blob_name}"

        response = requests.get(file_url, headers=headers)
        if response.status_code != 200:
            print(f"❌ Failed to download {blob_name}. Error: {response.status_code} | {response.text}")
            return None

        return response.json()


class VideoAnnotationLoader:
    """Handles loading videos and matching them with annotations."""

    def __init__(self, storage_handler):
        self.storage_handler = storage_handler

    def load_videos_with_annotations(self):
        """Loads videos from GCS into memory and pairs them with annotations."""
        video_files = self.storage_handler.list_files(file_extension=".mp4")
        annotation_files = self.storage_handler.list_files(prefix=ANNOTATIONS_PATH, file_extension=".json")
        combined_data = {}

        for video in map(lambda v: v.decode('utf-8') if isinstance(v, bytes) else v, video_files):
            video_name = os.path.basename(video).replace(".mp4", "")

            matching_annotation = next(
                (ann for ann in annotation_files if os.path.basename(ann).replace(".json", "") == video_name), None
            )

            video_stream = self.storage_handler.stream_video(video)
            annotation_data = self.storage_handler.download_json(matching_annotation) if matching_annotation else None

            if video_stream and annotation_data:
                combined_data[video] = {
                    "video_stream": video_stream,
                    "annotation": annotation_data
                }
        return combined_data
