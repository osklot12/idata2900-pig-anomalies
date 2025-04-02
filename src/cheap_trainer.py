from src.network.messages.serialization.pickle_message_deserializer import PickleMessageDeserializer
from src.network.messages.serialization.pickle_message_serializer import PickleMessageSerializer
from src.models.twod.fastrcnn.train_fastrcnn import RCNNTrainer
from src.network.client.simple_network_client import SimpleNetworkClient
from src.network.network_frame_instance_provider import NetworkFrameInstanceProvider
from src.data.streaming.prefetchers.batch_prefetcher import BatchPrefetcher
from src.data.dataset.dataset_split import DatasetSplit
from src.ui.telemetry.rich_dashboard import RichDashboard


class TrainingPipelineCoordinator:
    def __init__(self, server_ip="10.0.0.1"):
        self.client = SimpleNetworkClient(
            serializer=PickleMessageSerializer(),
            deserializer=PickleMessageDeserializer()
        )
        self.server_ip = server_ip
        self.dashboard = RichDashboard()

    def start(self):
        print("[Coordinator] Launching dashboard...")
        self.dashboard.start()

        print("[Coordinator] Connecting to server...")
        self.client.connect(self.server_ip)

        # Set up provider + prefetcher
        provider = NetworkFrameInstanceProvider(self.client)
        prefetcher = BatchPrefetcher(
            batch_provider=provider,
            split=DatasetSplit.TRAIN,
            batch_size=8,
            buffer_size=4
        )
        prefetcher.run()

        print("[Coordinator] Launching training...")
        trainer = RCNNTrainer(prefetcher)
        trainer.train()

        print("[Coordinator] Training finished. Cleaning up.")
        prefetcher.stop()
        self.client.disconnect()


if __name__ == "__main__":
    coordinator = TrainingPipelineCoordinator()
    coordinator.start()
