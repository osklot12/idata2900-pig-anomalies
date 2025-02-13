import tensorflow as tf
import random
import threading
import json
from src.data.gcp_data_loader import GCPDataLoader
from src.data.image_augmentor import ImageAugmentor

class DataManager:
    """
    Manages dataset loading, seeding, shuffling, and ensures annotation consistency during augmentation.
    """

    def __init__(self, bucket_name, credentials_path, seed=42, shuffle_buffer_size=1000, num_threads=4):
        """
        Initializes the DataManager.

        :param bucket_name: Name of the GCP bucket.
        :param credentials_path: Path to GCP credentials.
        :param seed: Random seed for consistent dataset order.
        :param shuffle_buffer_size: Size of shuffle buffer to break temporal dependency.
        :param num_threads: Number of parallel threads for data loading.
        """
        self.gcp_loader = GCPDataLoader(bucket_name, credentials_path)
        self.augmentor = ImageAugmentor(target_size=(224, 224))
        self.seed = seed
        self.shuffle_buffer_size = shuffle_buffer_size
        self.num_threads = num_threads
        self.loaded_videos = set()  # Tracks already loaded videos to avoid duplicates
        self.lock = threading.Lock()  # Prevents race conditions in multithreading
        random.seed(self.seed)

    def load_frames_from_video(self, video_name):
        """
        Loads frames from a specific video.

        :param video_name: The name of the video file.
        :return: List of frame tensors and corresponding annotations.
        """
        frame_list = self.gcp_loader.list_files(prefix=f"frames/{video_name}/", file_extension=".jpg")
        annotation_file = f"annotations/{video_name}.json"
        annotation_data = self.gcp_loader.download_json(annotation_file) if annotation_file else None

        frames = []
        for frame_name in frame_list:
            frame = self.gcp_loader.stream_video(frame_name)
            frame_tensor = tf.image.decode_jpeg(frame.read(), channels=3)
            frame_tensor = tf.image.resize(frame_tensor, (224, 224)) / 255.0  # Normalize
            frames.append((frame_name, frame_tensor))

        return frames, annotation_data

    def process_video(self, video_name):
        """
        Loads, augments, and processes a video.

        :param video_name: The video to process.
        :return: Processed frames and updated annotations.
        """
        frames, annotations = self.load_frames_from_video(video_name)

        if not frames:
            print(f"❌ No frames found for {video_name}")
            return [], None

        processed_frames = []
        updated_annotations = {}

        for frame_name, frame in frames:
            augmented_frame, new_annotations = self.augment_frame_and_annotations(frame, frame_name, annotations)
            processed_frames.append((frame_name, augmented_frame))
            updated_annotations[frame_name] = new_annotations

        return processed_frames, updated_annotations

    def augment_frame_and_annotations(self, frame, frame_name, annotations):
        """
        Augments the frame and updates its corresponding annotations.

        :param frame: TensorFlow image tensor.
        :param frame_name: The filename of the frame.
        :param annotations: The annotation dictionary.
        :return: Augmented frame and updated annotations.
        """
        original_height, original_width, _ = frame.shape

        # Select random transformations (flip, rotate)
        flip_horizontal = random.choice([True, False])
        flip_vertical = random.choice([True, False])
        rotation_angle = random.choice([0, 90, 180, 270])

        augmented_frame = frame
        if flip_horizontal:
            augmented_frame = tf.image.flip_left_right(augmented_frame)
        if flip_vertical:
            augmented_frame = tf.image.flip_up_down(augmented_frame)
        if rotation_angle == 90:
            augmented_frame = tf.image.rot90(augmented_frame, k=1)
        elif rotation_angle == 180:
            augmented_frame = tf.image.rot90(augmented_frame, k=2)
        elif rotation_angle == 270:
            augmented_frame = tf.image.rot90(augmented_frame, k=3)

        # Get bounding box for this frame (if exists)
        frame_index = int(frame_name.split("_")[-1].split(".")[0])  # Extract frame index from name
        bbox_data = annotations["annotations"][0]["frames"].get(str(frame_index), {}).get("bounding_box", None)

        if bbox_data:
            bbox_data = self.adjust_bbox(bbox_data, original_width, original_height, flip_horizontal, flip_vertical, rotation_angle)

        return augmented_frame, bbox_data

    def adjust_bbox(self, bbox, orig_width, orig_height, flip_horizontal, flip_vertical, rotation_angle):
        """
        Adjusts bounding box coordinates based on the augmentation applied.

        :param bbox: Original bounding box data.
        :param orig_width: Original image width.
        :param orig_height: Original image height.
        :param flip_horizontal: Whether the image was flipped horizontally.
        :param flip_vertical: Whether the image was flipped vertically.
        :param rotation_angle: Rotation applied (0, 90, 180, 270).
        :return: Adjusted bounding box.
        """
        x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]

        # Flip Adjustments
        if flip_horizontal:
            x = orig_width - (x + w)
        if flip_vertical:
            y = orig_height - (y + h)

        # Rotation Adjustments
        if rotation_angle == 90:
            x_new = y
            y_new = orig_width - (x + w)
            w_new = h
            h_new = w
        elif rotation_angle == 180:
            x_new = orig_width - (x + w)
            y_new = orig_height - (y + h)
            w_new = w
            h_new = h
        elif rotation_angle == 270:
            x_new = orig_height - (y + h)
            y_new = x
            w_new = h
            h_new = w
        else:  # No rotation
            x_new, y_new, w_new, h_new = x, y, w, h

        return {"x": x_new, "y": y_new, "w": w_new, "h": h_new}

