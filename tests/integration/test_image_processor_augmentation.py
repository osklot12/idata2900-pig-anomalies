import pytest
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from src.data.image_processor import ImageProcessor
from src.data.image_augmentor import ImageAugmentor

TEST_FRAMES_DIR = "../data/test_frames"

@pytest.fixture
def test_images():
    """Loads all test images as TensorFlow tensors."""
    images = []
    for filename in os.listdir(TEST_FRAMES_DIR):
        if filename.endswith(".png") or filename.endswith(".jpg"):  # Supports PNG & JPEG
            image_path = os.path.join(TEST_FRAMES_DIR, filename)
            image_bytes = tf.io.read_file(image_path)
            if filename.endswith(".png"):
                image_tensor = tf.image.decode_png(image_bytes, channels=3)
            else:
                image_tensor = tf.image.decode_jpeg(image_bytes, channels=3)
            images.append(image_tensor)
    return images


def test_processing_and_augmentation(test_images):
    """Processes an image and generates 10 highly diverse augmented versions, then displays them."""
    processor = ImageProcessor(target_size=(224, 224), norm_range=(0, 1))
    augmentor = ImageAugmentor(target_size=(224, 224))

    sample_image = test_images[0]  # Use the first image for testing

    # Step 1: Process (resize & normalize)
    processed_image = processor.process(sample_image)

    # Step 2: Augment (create 10 variations)
    augmented_images = [augmentor.process(tf.identity(processed_image)) for _ in range(10)]

    # Step 3: Display the images
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle("Augmented Images Randomized", fontsize=14)

    for idx, ax in enumerate(axes.flat):
        ax.imshow(augmented_images[idx].numpy())  # Convert tensor to NumPy array
        ax.axis("off")
        ax.set_title(f"Version {idx+1}")

    plt.show()
