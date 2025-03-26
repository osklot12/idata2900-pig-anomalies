import time

from src.data.pipeline.cheap_pipeline import CheapPipeline


def run():
    pipeline = CheapPipeline()
    try:
        pipeline.run()
        print("Pipeline is running. Press CTRL+C to stop.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping pipeline...")
        pipeline.stop()
        print("Pipeline stopped.")


if __name__ == "__main__":
    run()
