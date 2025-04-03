from yolox.core.trainer import Trainer

class StreamingTrainer(Trainer):
    """A custom trainer for streaming data."""

    def get_train_loader(self):
        return self.exp.get_data_loader(
            self.args.batch_size,
            is_distributed=False,
            no_aug=False
        )

    def get_eval_loader(self):
        return self.exp.get_eval_loader(
            self.args.batch_size,
            is_distributed=False
        )