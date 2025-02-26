class COCOFormatter:
    """
    Converts processed batches into COCO format.
    """
    def __init__(self):
        self.categories = {}  # Category name → ID mapping
        self.image_id = 1
        self.annotation_id = 1

    def process(self, batch_data):
        """
        Converts images and tuple-based annotations into COCO format.

        :param batch_data: List of (image_array, annotation_list) tuples.
        :return: COCO-formatted dataset (dict).
        """
        coco_data = {"images": [], "annotations": [], "categories": []}

        for image, annotations in batch_data:
            img_height, img_width = image.shape[:2]

            # Register image in COCO format
            coco_data["images"].append({
                "id": self.image_id,
                "file_name": f"frame_{self.image_id}.jpg",
                "width": img_width,
                "height": img_height
            })

            for cls, x, y, w, h in annotations:
                if cls not in self.categories:
                    self.categories[cls] = len(self.categories) + 1

                coco_data["annotations"].append({
                    "id": self.annotation_id,
                    "image_id": self.image_id,
                    "category_id": self.categories[cls],
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": 0
                })

                self.annotation_id += 1

            self.image_id += 1

        # Convert categories to COCO format
        coco_data["categories"] = [
            {"id": cid, "name": cname} for cname, cid in self.categories.items()
        ]

        return coco_data
