from multiprocessing import Process
from src.worker.pipeline_ddp_receiver import PipelineDataReceiver
from src.worker.rcnn_trainer import launch_training

class TrainingPipelineCoordinator:
    def __init__(self, world_size=1, host="0.0.0.0", port=50051):
        self.receiver = PipelineDataReceiver(host=host, port=port)
        self.world_size = world_size

    def start(self):
        print("[Coordinator] Starting receiver...")
        receiver_process = Process(target=self.receiver.start)
        receiver_process.start()

        print("[Coordinator] Launching training...")
        launch_training(self.world_size)

        receiver_process.join()
