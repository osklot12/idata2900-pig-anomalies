from src.data.streaming.prefetchers.fake_batch_prefetcher import FakeBatchPrefetcher
from src.network.client.simple_network_client import SimpleNetworkClient
from src.network.messages.serialization.pickle_message_deserializer import PickleMessageDeserializer
from src.network.messages.serialization.pickle_message_serializer import PickleMessageSerializer
from src.network.network_frame_instance_provider import NetworkFrameInstanceProvider
from src.models.twod.yolo.xi.yoloxi_dataset import UltralyticsDataset
from src.models.twod.yolo.xi.yoloxi_training_setup import TrainingSetup


def main():
    client = SimpleNetworkClient(PickleMessageSerializer(), PickleMessageDeserializer())
    client.connect("10.0.0.1")

    batch_provider = NetworkFrameInstanceProvider(client)
    prefetcher = FakeBatchPrefetcher(batch_size=8)
    prefetcher.run()
    dataset = UltralyticsDataset(prefetcher)

    setup = TrainingSetup(dataset=dataset)
    setup.train()


if __name__ == "__main__":
    main()
