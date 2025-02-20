import pytest
import tensorflow as tf
import json
import os
from src.data.image_augmentor import ImageAugmentor

# Paths to test frames and annotation file
TEST_FRAMES_DIR = "../data/test_frames/"
TEST_ANNOTATION_FILE = "../data/annotations/annotation_darwin.json"

@pytest.fixture
def annotation_data():
    """Loads the annotation JSON file."""
    with open(TEST_ANNOTATION_FILE, "r") as f:
        data = json.load(f)
    return data

@pytest.fixture
def augmentor():
    """Creates an ImageAugmentor instance."""
    return ImageAugmentor(target_size=(224, 224))

@pytest.mark.parametrize("frame_file, frame_index", [
    ("pigbelly-frame87.jpg", 87),
    ("pigbelly-frame151.jpg", 151)
])
def test_augment_specific_frames(annotation_data, augmentor, frame_file, frame_index):
    """Tests whether bounding boxes correctly transform with image augmentations for specific frames."""

    # Load the actual frame file
    frame_path = os.path.join(TEST_FRAMES_DIR, frame_file)
    assert os.path.exists(frame_path), f"❌ Frame file {frame_path} not found."

    image_bytes = tf.io.read_file(frame_path)
    frame_tensor = tf.image.decode_jpeg(image_bytes, channels=3)
    orig_height, orig_width, _ = frame_tensor.shape

    # Convert frame_index to string to match JSON keys
    frame_index_str = str(frame_index)

    # 🔍 Search for frame in ALL annotation sections
    bbox = None
    for annotation in annotation_data["annotations"]:
        if frame_index_str in annotation["frames"]:
            bbox = annotation["frames"][frame_index_str]["bounding_box"]
            break  # Stop searching once found

    # Debugging: Print all available annotated frames
    available_frames = set()
    for annotation in annotation_data["annotations"]:
        available_frames.update(annotation["frames"].keys())

    print(f"🔍 Available frames in annotations: {sorted(available_frames)}")

    assert bbox is not None, f"❌ No bounding box found for frame {frame_index} in annotations."

    print(f"✅ Original Bounding Box for Frame {frame_index}: {bbox}")

    # Apply augmentations
    flip_horizontal = tf.random.uniform([], 0, 1) > 0.5
    flip_vertical = tf.random.uniform([], 0, 1) > 0.5
    rotation_angle = tf.random.uniform([], 0, 4, dtype=tf.int32) * 90  # 0, 90, 180, 270
    brightness_adjust = tf.random.uniform([], -0.2, 0.2)
    contrast_adjust = tf.random.uniform([], 0.8, 1.2)

    augmented_frame = frame_tensor

    print(f"🔄 Augmentations Applied to Frame {frame_index}:")
    if flip_horizontal:
        augmented_frame = tf.image.flip_left_right(augmented_frame)
        print("   ➤ Flipped Horizontally ✅")
    if flip_vertical:
        augmented_frame = tf.image.flip_up_down(augmented_frame)
        print("   ➤ Flipped Vertically ✅")
    if rotation_angle == 90:
        augmented_frame = tf.image.rot90(augmented_frame, k=1)
        print("   ➤ Rotated 90° Left (Counterclockwise) ✅")
    elif rotation_angle == 180:
        augmented_frame = tf.image.rot90(augmented_frame, k=2)
        print("   ➤ Rotated 180° ✅")
    elif rotation_angle == 270:
        augmented_frame = tf.image.rot90(augmented_frame, k=3)
        print("   ➤ Rotated 90° Right (Clockwise) ✅")

    augmented_frame = tf.image.adjust_brightness(augmented_frame, brightness_adjust)
    print(f"   ➤ Brightness Adjusted by {brightness_adjust.numpy():.2f} ✅")

    augmented_frame = tf.image.adjust_contrast(augmented_frame, contrast_adjust)
    print(f"   ➤ Contrast Adjusted by {contrast_adjust.numpy():.2f} ✅")

    # Adjust bounding box
    augmented_bbox = adjust_bbox(bbox, orig_width, orig_height, flip_horizontal, flip_vertical, rotation_angle)

    # Validate Augmented Bounding Box
    aug_height, aug_width, _ = augmented_frame.shape

    assert 0 <= augmented_bbox["x"] <= aug_width, f"❌ x-coordinate out of bounds for frame {frame_index}: {augmented_bbox['x']}"
    assert 0 <= augmented_bbox["y"] <= aug_height, f"❌ y-coordinate out of bounds for frame {frame_index}: {augmented_bbox['y']}"
    assert 0 <= augmented_bbox["w"] <= aug_width, f"❌ Width out of bounds for frame {frame_index}: {augmented_bbox['w']}"
    assert 0 <= augmented_bbox["h"] <= aug_height, f"❌ Height out of bounds for frame {frame_index}: {augmented_bbox['h']}"

    print(f"✅ Augmented Bounding Box for Frame {frame_index}: {augmented_bbox}")
    print("=" * 80)  # Separator for readability

def adjust_bbox(bbox, orig_width, orig_height, flip_horizontal, flip_vertical, rotation_angle):
    """
    Adjusts bounding box coordinates based on applied augmentations.

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
