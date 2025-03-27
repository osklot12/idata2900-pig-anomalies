import time

from src.data.dataset_split import DatasetSplit
from src.data.pipeline.cheap_pipeline import CheapPipeline
from src.network.pipeline_ddp_sender import PipelineDataSender


def run():
    pipeline = CheapPipeline()
    server = PipelineDataSender(["10.0.0.2"])
    server.connect_to_workers()
    try:
        print("Pipeline is running.")
        pipeline.run()
        while True:
            time.sleep(1)
            batch_size = 10
            split_size = pipeline._virtual_dataset.get_frame_count(DatasetSplit.TEST)
            while split_size < batch_size:
                time.sleep(1)
            server.send_data(
                pipeline.get_batch(DatasetSplit.TEST, batch_size)
            )
    except KeyboardInterrupt:
        print("Stopping pipeline...")
        pipeline.stop()
        print("Pipeline stopped.")


if __name__ == "__main__":
    run()
