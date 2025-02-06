import numpy as np
import matplotlib.pyplot as plt
import imageio
from io import BytesIO
from src.data.data_loader import load_videos_with_annotations


def load_videos_sequentially():
    print("🚀 Running sequential video loading test...")
    video_queue = load_videos_with_annotations()

    if not video_queue:
        print("❌ No videos found in the bucket.")
        return

    video_keys = sorted(video_queue.keys())  # Ensure sorted order

    for video_key in video_keys:
        video_stream = video_queue[video_key]["video_stream"]

        if not isinstance(video_stream, BytesIO):
            print(f"❌ video_stream for {video_key} is not a valid BytesIO object.")
            continue

        # Check video size
        video_stream.seek(0, 2)  # Move to end
        video_size = video_stream.tell()
        video_stream.seek(0)  # Reset position
        print(f"📏 Video stream size for {video_key}: {video_size} bytes")

        if video_size == 0:
            print(f"❌ The video stream for {video_key} is empty.")
            continue

        print(f"📽️ Processing video: {video_key}")
        yield video_key, video_stream


def test_video_frame_extraction():
    for video_key, video_stream in load_videos_sequentially():
        print(f"📽️ Extracting a single frame from {video_key}...")
        try:
            reader = imageio.get_reader(video_stream, format='mp4')
            frame = reader.get_next_data()
            reader.close()

            # Display extracted frame
            plt.imshow(frame)
            plt.axis("off")
            plt.title(f"Extracted Frame from {video_key}")
            plt.show()
            print("✅ Frame extracted successfully.")
        except Exception as e:
            print(f"❌ Failed to extract frame from {video_key}: {e}")


# Run the sequential loader
test_video_frame_extraction()
