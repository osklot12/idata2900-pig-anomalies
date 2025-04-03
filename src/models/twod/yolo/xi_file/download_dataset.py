import os
import shutil
import time
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
        angle = 0
        label_lines.append(f"{ann.cls.value} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} {angle:.6f}")

    with open(out_label_dir / label_filename, "w") as f:
        f.write("\n".join(label_lines))


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
    prefetcher = BatchPrefetcher(batch_provider, SPLIT, BATCH_SIZE)
    prefetcher.run()

    # Start pulling batches and saving
    image_idx = 0
    for _ in range(NUM_BATCHES):
        time.sleep(.2)
        batch = prefetcher.get()
        for frame in batch:
            save_yolo_obb_format(
                frame,
                OUT_DIR / "images" / "train",
                OUT_DIR / "labels" / "train",
                image_idx
            )
            image_idx += 1

    # Save YOLO data.yaml
    with open(OUT_DIR / "data.yaml", "w") as f:
        f.write(f"""\
train: {OUT_DIR / 'images' / 'train'}
val: {OUT_DIR / 'images' / 'train'}  # Using same for now
nc: 4
names: ['tail-biting', 'belly-nosing', 'ear-biting', 'tail-down']
obb: True
""")

    print(f"✅ Download complete: {image_idx} images saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
