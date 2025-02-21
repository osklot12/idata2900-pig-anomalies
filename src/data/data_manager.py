import tensorflow as tf
import json
import os
from src.data.image_augmentor import ImageAugmentor

class DataManager:
    """
    Manages dataset loading, augmentation, and ensures annotation consistency.
    """

    def __init__(self, output_dir, num_augmented_versions=10):
        self.augmentor = ImageAugmentor(target_size=(224, 224))
        self.output_dir = output_dir
        self.num_augmented_versions = num_augmented_versions
        self.memory_store = []  # Store processed data in memory

    def process_video_frames(self, video_name, frame_data, annotation_data):
        """
        Handles the full data processing pipeline for a single video.
        """

        assert frame_data, f"❌ No frame data provided for {video_name}"
        assert annotation_data, f"❌ No annotation data provided for {video_name}"

        # Process and store in memory
        for frame_name, frame_tensor in frame_data.items():
            frame_idx = int(frame_name.split("_")[1])  # Extract frame index
            original_annotation = annotation_data["annotations"][0]["frames"].get(str(frame_idx), {})

            for aug_version in range(self.num_augmented_versions):
                # Pass both image and annotation to ImageAugmentor
                augmented_frame, augmented_annotation = self.augmentor.process(frame_tensor, original_annotation)

                # 🔹 Ensure annotations are JSON-compatible
                safe_annotation = self._convert_annotations(augmented_annotation)

                # 🔹 Store in memory
                self.memory_store.append({
                    "frame": tf.convert_to_tensor(augmented_frame),
                    "annotation": json.dumps(safe_annotation)
                })


    def _convert_annotations(self, obj):
        """
        Recursively converts TensorFlow Tensors to standard Python types
        to ensure JSON serialization works.
        """
        if isinstance(obj, tf.Tensor):
            if obj.dtype == tf.string:
                return obj.numpy().decode('utf-8')
            else:
                return obj.numpy().tolist()  # Convert Tensor to Python type
        elif isinstance(obj, dict):
            return {key: self._convert_annotations(value) for key, value in obj.items()}  # Recursively convert dicts
        elif isinstance(obj, list):
            return [self._convert_annotations(item) for item in obj]  # Convert lists
        elif isinstance(obj, bytes):
            return obj.decode('utf-8')
        else:
            return obj  # Return original if no conversion needed
