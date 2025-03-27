from multiprocessing import Process

from src.worker.rcnn_eval import launch_evaluation
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
        launch_training(self.world_size, self.receiver.get_queue())

        receiver_process.join()

class EvaluationPipelineCoordinator:
    def __init__(self, world_size=1, host="0.0.0.0", port=50051):
        self.receiver = PipelineDataReceiver(host=host, port=port)
        self.world_size = world_size

    def start(self):
        print("[Coordinator] Starting receiver...")
        receiver_process = Process(target=self.receiver.start)
        receiver_process.start()

        print("[Coordinator] Launching evaluation...")
        launch_evaluation(world_size=1, queue=self.receiver.get_queue(), model_path="checkpoints/model_checkpoint_1000.pt")

        receiver_process.join()

if __name__ == "__main__":
    coordinator = EvaluationPipelineCoordinator()
    coordinator.start()
