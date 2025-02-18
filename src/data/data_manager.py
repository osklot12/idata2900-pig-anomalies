import tensorflow as tf
import json
import os
import random
from src.data.image_augmentor import ImageAugmentor

class DataManager:
    """
    Manages dataset loading, augmentation, and ensures annotation consistency.
    """

    def __init__(self, output_dir, num_augmented_versions=10):
        """
        Initializes the DataManager.

        :param output_dir: Directory to store processed frames and JSON annotations.
        :param num_augmented_versions: Number of augmented versions to generate per frame.
        """
        self.augmentor = ImageAugmentor(target_size=(224, 224))
        self.output_dir = output_dir
        self.num_augmented_versions = num_augmented_versions

    def process_video_frames(self, video_name, frame_data, annotation_data):
        """
        Handles the full data processing pipeline for a single video based on its extracted JPG frames.
        """
        assert frame_data, f"❌ No frame data provided for {video_name}"
        assert annotation_data, f"❌ No annotation data provided for {video_name}"

        # Organize augmented sequences
        augmented_sequences = {i: [] for i in range(self.num_augmented_versions)}
        annotation_sequences = {i: {} for i in range(self.num_augmented_versions)}

        for frame_name, frame_tensor in frame_data.items():
            frame_idx = int(frame_name.split("_")[1])  # Extract frame index from name
            original_annotation = annotation_data["annotations"][0]["frames"].get(str(frame_idx), {})

            for aug_version in range(self.num_augmented_versions):
                # Pass both image and annotation to ImageAugmentor
                augmented_frame, augmented_annotation = self.augmentor.process(frame_tensor, original_annotation)

                augmented_sequences[aug_version].append(augmented_frame)
                annotation_sequences[aug_version][frame_idx] = augmented_annotation

        self.save_augmented_data(video_name, augmented_sequences, annotation_sequences)

    def save_augmented_data(self, video_name, augmented_sequences, annotation_sequences):
        """
        Saves augmented frames and corresponding annotations.
        """
        for version in range(self.num_augmented_versions):
            version_dir = os.path.join(self.output_dir, f"{video_name}_version{version}")
            os.makedirs(version_dir, exist_ok=True)

            # Save frames
            for frame_idx, frame in enumerate(augmented_sequences[version]):
                frame_path = os.path.join(version_dir, f"frame_{frame_idx}.jpg")
                tf.keras.utils.save_img(frame_path, frame.numpy())

            # Convert EagerTensors in annotations to regular Python floats
            def convert_annotations(obj):
                if isinstance(obj, tf.Tensor):
                    return float(obj.numpy())  # Convert Tensor to float
                elif isinstance(obj, dict):
                    return {key: convert_annotations(value) for key, value in obj.items()}  # Recursively convert dicts
                elif isinstance(obj, list):
                    return [convert_annotations(item) for item in obj]  # Convert lists
                else:
                    return obj  # Return original if no conversion needed

            # Apply conversion to annotation JSON
            json_annotations = convert_annotations(annotation_sequences[version])

            # Save annotations
            annotation_path = os.path.join(version_dir, "annotations.json")
            with open(annotation_path, "w") as f:
                json.dump(json_annotations, f)
