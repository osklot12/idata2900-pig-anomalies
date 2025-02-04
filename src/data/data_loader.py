import os
import json
import requests
import tensorflow as tf
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from io import BytesIO

# 🔹 CONFIGURE YOUR BUCKET
BUCKET_NAME = "norsvin-g2b-behavior-prediction"
script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the script
CREDENTIALS_FILE = os.path.join(script_dir, "user_oauth_credentials.json")  # Absolute path
ANNOTATIONS_PATH = "g2b_behaviour/releases/g2b-prediction/annotations"
SEED = 42  # Ensures deterministic shuffling

SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]


def get_access_token():
    """Authenticate using stored OAuth token instead of client_secret.json."""
    creds = None

    if os.path.exists(CREDENTIALS_FILE):
        creds = Credentials.from_authorized_user_file(CREDENTIALS_FILE, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            raise FileNotFoundError("❌ OAuth token expired or missing. Ensure 'user_oauth_credentials.json' exists.")

    return creds.token


def list_files_in_bucket():
    """List all video files in the GCS bucket."""
    access_token = get_access_token()
    headers = {"Authorization": f"Bearer {access_token}"}
    url = f"https://www.googleapis.com/storage/v1/b/{BUCKET_NAME}/o"

    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print(f"❌ Error listing files: {response.text}")
        return []

    files = [file["name"] for file in response.json().get("items", []) if file["name"].endswith(".mp4")]
    print(f"🔍 Found {len(files)} videos in the bucket")
    print(f"📝 Sample filenames: {files[:3]}")
    files.sort()

    tf.random.set_seed(SEED)
    shuffled_files = tf.random.shuffle(files).numpy().tolist()
    return shuffled_files


def stream_video_from_gcs(blob_name):
    """Stream a video file from Google Cloud Storage into memory."""
    access_token = get_access_token()
    headers = {"Authorization": f"Bearer {access_token}"}
    file_url = f"https://storage.googleapis.com/{BUCKET_NAME}/{blob_name}"

    response = requests.get(file_url, headers=headers, stream=True, timeout=10)
    print(f"📥 Streaming {blob_name} | Status Code: {response.status_code}")
    if response.status_code != 200:
        print(f"❌ Failed to stream {blob_name}. Error: {response.status_code} | {response.text}")
    if response.status_code != 200:
        print(f"❌ Failed to stream {blob_name}. Error: {response.status_code} | {response.text}")
        return None

    return BytesIO(response.content)


def list_annotations():
    """List all annotation files in the annotations directory."""
    access_token = get_access_token()
    headers = {"Authorization": f"Bearer {access_token}"}
    url = f"https://www.googleapis.com/storage/v1/b/{BUCKET_NAME}/o?prefix={ANNOTATIONS_PATH}"

    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print(f"❌ Error listing annotation files: {response.text}")
        return []

    annotation_files = [file["name"] for file in response.json().get("items", []) if file["name"].endswith(".json")]

    print(f"📜 Retrieved {len(annotation_files)} annotations")
    print(f"📝 Sample annotations: {annotation_files[:5]}")  # Show first 5
    annotation_files.sort()
    return annotation_files



def download_json(blob_name):
    """Download an annotation JSON file from GCS and load it into memory."""
    access_token = get_access_token()
    headers = {"Authorization": f"Bearer {access_token}"}
    file_url = f"https://storage.googleapis.com/{BUCKET_NAME}/{blob_name}"

    response = requests.get(file_url, headers=headers)
    if response.status_code != 200:
        print(f"❌ Failed to download {blob_name}. Error: {response.status_code} | {response.text}")
        return None

    return response.json()


def load_videos_with_annotations():
    """Load videos from GCS into memory and pair them with corresponding annotations."""
    video_files = list_files_in_bucket()
    annotation_files = list_annotations()
    combined_data = {}

    for video in map(lambda v: v.decode('utf-8') if isinstance(v, bytes) else v, video_files):
        video_name = os.path.basename(video).replace(".mp4", "") if isinstance(video, str) else os.path.basename(
            video.decode('utf-8')).replace(".mp4", "")

        print(f"🧐 Extracted video filename: {video_name}")
        print(
            f"📂 Available annotation filenames: {[os.path.basename(a) for a in annotation_files][:5]}")  # Print first 5

        matching_annotation = next(
            (ann for ann in annotation_files if os.path.basename(ann).replace(".json", "") == video_name), None)

        print(f"🔎 Checking annotation for {video_name}")
        print(f"📄 Matching Annotation: {matching_annotation if matching_annotation else '❌ No match found'}")

        video_stream = stream_video_from_gcs(video)
        annotation_data = download_json(matching_annotation) if matching_annotation else None

        if video_stream and annotation_data:
            combined_data[video] = {
                "video_stream": video_stream,
                "annotation": annotation_data
            }

    print(f"✅ Total Videos Processed: {len(combined_data)}")
    for key, value in combined_data.items():
        print(f"🎥 Video: {key}")
        print(f"📝 Annotation Sample: {json.dumps(value['annotation'])[:300]}...")
    return combined_data


# Example usage
print("🚀 Streaming videos and loading annotations into memory...")
data = load_videos_with_annotations()
print("🎯 Videos and annotations loaded into memory!")
