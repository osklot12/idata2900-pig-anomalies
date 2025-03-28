from multiprocessing import Process

from src.network.messages.serialization.pickle_message_deserializer import PickleMessageDeserializer
from src.network.messages.serialization.pickle_message_serializer import PickleMessageSerializer
from src.worker.pipeline_ddp_receiver import PipelineDataReceiver
from src.models.twod.fastrcnn.train_fastrcnn import RCNNTrainer
from src.network.client.network_client import NetworkClient


class TrainingPipelineCoordinator:
    def __init__(self, host="0.0.0.0", port=50051):
        self.receiver = PipelineDataReceiver(host=host, port=port)

    def start(self):
        print("[Coordinator] Starting receiver...")
        receiver_process = Process(target=self.receiver.start)
        receiver_process.start()

        print("[Coordinator] Launching training...")
        client = NetworkClient("10.0.0.1", PickleMessageSerializer(), PickleMessageDeserializer())
        trainer = RCNNTrainer(client)
        trainer.train()

        receiver_process.join()


if __name__ == "__main__":
    coordinator = TrainingPipelineCoordinator()
    coordinator.start()