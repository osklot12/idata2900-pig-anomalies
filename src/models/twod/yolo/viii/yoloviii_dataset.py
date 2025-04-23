from torch.utils.data import IterableDataset
from src.data.dataset.streams.prefetcher import Prefetcher
from src.models.converters.viii.yoloviii_batch_converter import YOLOv8BatchConverter


class YOLOv8StreamingDataset(IterableDataset):
    def __init__(self, stream_factory, batch_size, max_batches, eval_mode=False):
        self.stream = stream_factory.create_stream()
        self.batch_size = batch_size
        self.max_batches = max_batches
        self.eval_mode = eval_mode
        self.stream.start()

    def __iter__(self):
        batch = []
        count = 0

        for frame in self.stream:
            if not frame.annotations:
                continue

            batch.append(frame)

            if len(batch) == self.batch_size:
                # ✅ Convert once we’ve collected a full batch
                images, targets = YOLOv8BatchConverter.convert(batch)
                yield images, targets

                batch = []
                count += 1
                if count >= self.max_batches:
                    break

        # Handle final batch if smaller than batch_size
        if batch:
            images, targets = YOLOv8BatchConverter.convert(batch)
            yield images, targets
