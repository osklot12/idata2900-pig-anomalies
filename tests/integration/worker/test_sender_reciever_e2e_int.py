import numpy as np
import time
import threading
from src.worker.pipeline_ddp_receiver import PipelineDataReceiver
from src.network.pipeline_ddp_sender import PipelineDataSender
from src.worker.data_queue import data_queue

def test_sender_receiver_pipeline():
    # --- Step 1: Start receiver in a thread (so it shares memory/queue) ---
    def run_receiver():
        receiver = PipelineDataReceiver(host="127.0.0.1", port=60000)
        receiver.start()

    receiver_thread = threading.Thread(target=run_receiver, daemon=True)
    receiver_thread.start()

    time.sleep(1)  # Let receiver start

    # --- Step 2: Connect and send using the real sender ---
    sender = PipelineDataSender(["127.0.0.1"], port=60000)
    sender.connect_to_workers()

    frame = (255 * np.ones((224, 224, 3))).astype(np.uint8)
    annotations = [[{"bbox": [5, 10, 100, 150], "label": 3}]]

    sender.send_data([frame], annotations)

    # --- Step 3: Give time to process the data ---
    time.sleep(1.5)

    # --- Step 4: Validate ---
    assert not data_queue.empty()
    image_tensor, target = data_queue.get()
    assert image_tensor.shape == (3, 224, 224)
    assert target["boxes"].shape == (1, 4)
    assert target["labels"].shape == (1,)
