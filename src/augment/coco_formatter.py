import json
import tensorflow as tf


class COCOFormatter:
    """Converts in-memory frames and annotations into COCO format."""

    def __init__(self):
        self.categories = {}  # Category name → ID mapping
        self.image_id = 1
        self.annotation_id = 1

    def process(self, augmented_data):
        """
        Processes augmented frames and annotations in memory, converting them to COCO format.

        :param augmented_data: List of dictionaries with {"frame": tf.Tensor, "annotation": dict}
        :return: COCO-formatted dataset (dict)
        """
        coco_data = {"images": [], "annotations": [], "categories": []}

        for entry in augmented_data:
            frame = entry["frame"]  # Augmented image tensor
            annotation = entry["annotation"]  # Processed annotation dict

            img_height, img_width = frame.shape[:2]

            # Register image in COCO format
            coco_data["images"].append({
                "id": self.image_id,
                "file_name": f"augmented_frame_{self.image_id}.jpg",
                "width": img_width,
                "height": img_height
            })

            # Extract bounding boxes from the correct structure
            for frame_id, frame_data in annotation.items():
                if "bounding_box" in frame_data:
                    bbox = frame_data["bounding_box"]
                    category_name = frame_data.get("text", {}).get("text", "unknown")

                    if category_name not in self.categories:
                        self.categories[category_name] = len(self.categories) + 1

                    # Add annotation
                    coco_data["annotations"].append({
                        "id": self.annotation_id,
                        "image_id": self.image_id,
                        "category_id": self.categories[category_name],
                        "bbox": [bbox["x"], bbox["y"], bbox["w"], bbox["h"]],
                        "area": bbox["w"] * bbox["h"],
                        "iscrowd": 0
                    })

                    self.annotation_id += 1

            self.image_id += 1

        # Convert categories to COCO format
        coco_data["categories"] = [
            {"id": cid, "name": cname} for cname, cid in self.categories.items()
        ]

        return coco_data  # Return COCO data in memory
