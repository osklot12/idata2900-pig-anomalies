from src.data.dataset.dataset_split import DatasetSplit
from src.data.streaming.prefetchers.batch_prefetcher import BatchPrefetcher
from src.models.twod.yolo.xi.yoloxi_dataset import UltralyticsDataset
from src.models.twod.yolo.xi.yoloxi_eval_dataset import EvalUltralyticsDataset
from src.network.client.simple_network_client import SimpleNetworkClient
from src.network.messages.serialization.pickle_message_deserializer import PickleMessageDeserializer
from src.network.messages.serialization.pickle_message_serializer import PickleMessageSerializer
from src.network.network_dataset_stream import NetworkDatasetStream

from time import sleep


def main():
    server_ip = "10.0.0.1"

    print("🔌 Connecting to train and val dataset streams...")
    train_client = SimpleNetworkClient(PickleMessageSerializer(), PickleMessageDeserializer())
    train_client.connect(server_ip)

    val_client = SimpleNetworkClient(PickleMessageSerializer(), PickleMessageDeserializer())
    val_client.connect(server_ip)

    train_stream = NetworkDatasetStream(client=train_client, split=DatasetSplit.TRAIN)
    val_stream = NetworkDatasetStream(client=val_client, split=DatasetSplit.VAL)

    batch_size = 8
    train_prefetcher = BatchPrefetcher(train_stream, batch_size=batch_size, fetch_timeout=300)
    val_prefetcher = BatchPrefetcher(val_stream, batch_size=batch_size, fetch_timeout=300)

    train_prefetcher.run()
    val_prefetcher.run()

    print("🚀 Streaming started: Train and Val prefetchers are now running")


    while(True):
        print(train_prefetcher.get())
        print(val_prefetcher.get())


if __name__ == "__main__":
    main()
