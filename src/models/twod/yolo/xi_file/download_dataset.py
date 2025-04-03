import os
import shutil
import time
import queue
from pathlib import Path
from PIL import Image

from src.data.dataset.dataset_split import DatasetSplit
from src.data.dataclasses.annotated_frame import AnnotatedFrame
from src.data.streaming.prefetchers.batch_prefetcher import BatchPrefetcher
from src.network.client.simple_network_client import SimpleNetworkClient
from src.network.messages.serialization.pickle_message_deserializer import PickleMessageDeserializer
from src.network.messages.serialization.pickle_message_serializer import PickleMessageSerializer
from src.network.network_frame_instance_provider import NetworkFrameInstanceProvider

# ========= CONFIG =========
OUT_DIR = Path("tmp_dataset")
NUM_BATCHES = 100
BATCH_SIZE = 8
SPLIT = DatasetSplit.TRAIN
SERVER_IP = "10.0.0.1"
# ==========================


def save_yolo_obb_format(frame: AnnotatedFrame, out_img_dir: Path, out_label_dir: Path, image_idx: int):
    img = Image.fromarray(frame.frame)
    img_filename = f"{image_idx:06d}.jpg"
    label_filename = f"{image_idx:06d}.txt"

    img.save(out_img_dir / img_filename)

    h, w = img.height, img.width
    label_lines = []
    for ann in frame.annotations:
        cx = (ann.bbox.x + ann.bbox.width / 2) / w
        cy = (ann.bbox.y + ann.bbox.height / 2) / h
        bw = ann.bbox.width / w
        bh = ann.bbox.height / h
        angle = 0  # Replace with actual angle if needed
        label_lines.append(f"{ann.cls.value} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} {angle:.6f}")

    with open(out_label_dir / label_filename, "w") as f:
        f.write("\n".join(label_lines))


def save_data_yaml(image_dir: Path, label_dir: Path):
    yaml_path = OUT_DIR / "data.yaml"
    with open(yaml_path, "w") as f:
        f.write(f"""\
train: {image_dir}
val: {image_dir}  # Using same for now
nc: 4
names: ['tail-biting', 'belly-nosing', 'ear-biting', 'tail-down']
obb: True
""")
    print(f"✅ Wrote data.yaml to {yaml_path}")


def main():
    # Clean old data
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    (OUT_DIR / "images" / "train").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "labels" / "train").mkdir(parents=True, exist_ok=True)

    # Set up network client and prefetcher
    client = SimpleNetworkClient(PickleMessageSerializer(), PickleMessageDeserializer())
    client.connect(SERVER_IP)
    batch_provider = NetworkFrameInstanceProvider(client)
    prefetcher = BatchPrefetcher(batch_provider, SPLIT, BATCH_SIZE, fetch_timeout=30)
    prefetcher.run()

    image_idx = 0

    try:
        for batch_num in range(NUM_BATCHES):
            try:
                time.sleep(0.2)
                batch = prefetcher.get()
            except queue.Empty:
                print(f"[Batch {batch_num}] ⚠️ Timed out waiting for batch. Stopping early.")
                break
            except Exception as e:
                print(f"[Batch {batch_num}] ❌ Unexpected error: {e}")
                break

            for frame in batch:
                try:
                    save_yolo_obb_format(
                        frame,
                        OUT_DIR / "images" / "train",
                        OUT_DIR / "labels" / "train",
                        image_idx
                    )
                    image_idx += 1
                except Exception as e:
                    print(f"[Image {image_idx}] ❌ Failed to save: {e}")

    finally:
        prefetcher.stop()
        save_data_yaml(OUT_DIR / "images" / "train", OUT_DIR / "labels" / "train")
        print(f"✅ Download complete. {image_idx} images saved in {OUT_DIR}")


if __name__ == "__main__":
    main()
