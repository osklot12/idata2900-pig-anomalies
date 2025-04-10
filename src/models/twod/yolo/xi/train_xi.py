from src.data.dataset.dataset_split import DatasetSplit
from src.data.streaming.prefetchers.batch_prefetcher import BatchPrefetcher
from src.data.streaming.prefetchers.fake_batch_prefetcher import FakeBatchPrefetcher
from src.models.twod.yolo.xi.yoloxi_dataset import UltralyticsDataset
from src.models.twod.yolo.xi.yoloxi_eval_dataset import EvalUltralyticsDataset
from src.models.twod.yolo.xi.yoloxi_training_setup import TrainingSetup
from src.network.client.simple_network_client import SimpleNetworkClient
from src.network.messages.serialization.pickle_message_deserializer import PickleMessageDeserializer
from src.network.messages.serialization.pickle_message_serializer import PickleMessageSerializer
from src.network.network_dataset_stream import NetworkDatasetStream


def main():
    server_ip = "10.0.0.1"

    train_client = SimpleNetworkClient(PickleMessageSerializer(), PickleMessageDeserializer())
    train_client.connect(server_ip)

    val_client = SimpleNetworkClient(PickleMessageSerializer(), PickleMessageDeserializer())
    val_client.connect(server_ip)

    train_stream = NetworkDatasetStream(client=train_client, split=DatasetSplit.TRAIN)
    val_stream = NetworkDatasetStream(client=val_client, split=DatasetSplit.VAL)

    batch_size = 8
    train_prefetcher = BatchPrefetcher(train_stream, batch_size=batch_size, fetch_timeout=200)
    val_prefetcher = BatchPrefetcher(val_stream, batch_size=batch_size, fetch_timeout=200)

    train_prefetcher.run()
    val_prefetcher.run()

    train_dataset = UltralyticsDataset(train_prefetcher, 6495)
    val_dataset = EvalUltralyticsDataset(val_prefetcher, 6495)

    trainer = TrainingSetup(train_dataset, val_dataset)
    trainer.train()



if __name__ == "__main__":
    main()
