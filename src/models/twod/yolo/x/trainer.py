from yolox.core.trainer import Trainer

class StreamingTrainer(Trainer):
    """Custom trainer class for YOLOX."""

    def get_train_loader(self):
        print("get_train_loader() called")
        return self.exp.get_data_loader(
            self.args.batch_size,
            is_distributed=False,
            no_aug=False
        )

    def get_eval_loader(self):
        print("get_eval_loader() called")
        return self.exp.get_eval_loader(
            self.args.batch_size,
            is_distributed=False,
        )