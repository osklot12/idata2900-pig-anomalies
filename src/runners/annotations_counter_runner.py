import json
from collections import defaultdict

from src.utils.path_finder import PathFinder


def main():
    # Path to your JSON file
    json_path = PathFinder.get_abs_path("cache/metadata.json")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    class_counts = defaultdict(int)

    for group in data.get("0"):
        for video_annotations in group.values():
            for behavior_class, count in video_annotations.items():
                class_counts[behavior_class] += count

    # Display results
    print("Annotation counts by class:")
    for behavior_class, count in class_counts.items():
        print(f"{behavior_class}: {count}")

if __name__ == "__main__":
    main()