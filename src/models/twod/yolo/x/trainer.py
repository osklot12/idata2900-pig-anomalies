from yolox.core.trainer import Trainer

class StreamingTrainer(Trainer):
    """Custom trainer class for YOLOX."""

    def train(self):
        print("Custom training function called")

        train_loader = self.exp.get_data_loader(
            self.args.batch_size,
            is_distributed=False,
            no_aug=True
        )

        for batch in train_loader:
            print("Got batch: ", batch)
            break