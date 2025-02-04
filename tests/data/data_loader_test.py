import matplotlib.pyplot as plt
import imageio
from io import BytesIO
from src.data.data_loader import load_videos_with_annotations


def test_video_frame_extraction():
    print("🚀 Running video frame extraction test...")
    data = load_videos_with_annotations()

    if not data:
        print("❌ No videos loaded.")
        return

    video_key = list(data.keys())[0]  # Get the first video
    video_stream = data[video_key]["video_stream"]

    if not isinstance(video_stream, BytesIO):
        print("❌ video_stream is not a valid BytesIO object.")
        return

    # Check video size
    video_stream.seek(0, 2)  # Move to end
    video_size = video_stream.tell()
    video_stream.seek(0)  # Reset position
    print(f"📏 Video stream size: {video_size} bytes")

    if video_size == 0:
        print("❌ The video stream is empty.")
        return

    print("📽️ Extracting a single frame...")
    try:
        # Read video and extract first frame
        reader = imageio.get_reader(video_stream, format='mp4')
        frame = reader.get_next_data()
        reader.close()

        # Display extracted frame
        plt.imshow(frame)
        plt.axis("off")
        plt.title("Extracted Frame from Video")
        plt.show()
        print("✅ Frame extracted successfully.")
    except Exception as e:
        print(f"❌ Failed to extract frame: {e}")


# Run the test
test_video_frame_extraction()
