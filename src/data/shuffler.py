import random
import json

class Shuffler:
    """Shuffles frames from multiple videos while ensuring each frame's annotation stays linked."""

    def __init__(self, seed=None):
        self.seed = seed
        if self.seed:
            random.seed(self.seed)

    def shuffle(self, frames, annotations):
        """
        Shuffles frames from multiple videos while preserving annotation links.
        Outputs:
        - A shuffled list of frames.
        - A single merged JSON annotation file that aligns with the shuffled frames.

        :param frames: List of dictionaries with {"video_id": str, "frame": tf.Tensor, "frame_idx": int}
        :param annotations: Dict mapping video IDs to their corresponding annotation JSON data.
        :return: (shuffled_frames, merged_annotations_json)
        """
        # Shuffle frames while keeping their original video ID reference
        random.shuffle(frames)

        # Initialize merged annotation JSON structure
        merged_annotations = {
            "images": [],
            "annotations": [],
            "categories": [],
        }
        category_map = {}  # Mapping of category names to unique IDs
        image_id = 1
        annotation_id = 1

        shuffled_data = []
        for frame_entry in frames:
            video_id = frame_entry["video_id"]
            frame_idx = frame_entry["frame_idx"]

            # Extract the correct annotation from the corresponding video's JSON
            frame_annotation = annotations.get(video_id, {}).get("annotations", [{}])[0].get("frames", {}).get(str(frame_idx), {})

            # Ensure "bounding_box" key exists
            bounding_box = frame_annotation.get("bounding_box", {"x": 0, "y": 0, "w": 0, "h": 0})
            category_name = frame_annotation.get("text", {}).get("text", "unknown")

            # Assign category IDs dynamically
            if category_name not in category_map:
                category_map[category_name] = len(category_map) + 1

            shuffled_data.append({
                "video_id": video_id,
                "frame": frame_entry["frame"],
                "frame_idx": frame_idx,
                "annotation": {
                    "bounding_box": bounding_box,
                    "category_id": category_map[category_name]
                }
            })

            # Add frame info to COCO-style merged annotation
            merged_annotations["images"].append({
                "id": image_id,
                "file_name": f"shuffled_frame_{image_id}.jpg",
                "width": 224,
                "height": 224
            })

            # Add annotation info
            merged_annotations["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_map[category_name],
                "bbox": [bounding_box["x"], bounding_box["y"], bounding_box["w"], bounding_box["h"]],
                "area": bounding_box["w"] * bounding_box["h"],
                "iscrowd": 0
            })

            image_id += 1
            annotation_id += 1

        # Convert categories to COCO format
        merged_annotations["categories"] = [{"id": cid, "name": cname} for cname, cid in category_map.items()]

        return shuffled_data, merged_annotations
