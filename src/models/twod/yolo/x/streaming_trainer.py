from typing import Dict

import torch

from src.models.twod.yolo.x.streaming_evaluator import StreamingEvaluator
from yolox.core.trainer import Trainer
from yolox.utils import is_parallel, adjust_status, synchronize


class StreamingTrainer(Trainer):
    """A custom trainer for streaming data."""


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.iter_loader = None

    def train_in_epoch(self):
        """Train the model for one epoch."""
        self.iter_loader = iter(self.get_train_loader())

        self.model.train()
        self.before_epoch()

        for self.iter in range(self.iters_per_epoch):
            self.before_iter()
            self.train_one_iter()
            self.after_iter()

        self.after_epoch()

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

    def evaluate_and_save_model(self):
        if self.use_model_ema:
            eval_model = self.ema_model.ema
        else:
            eval_model = self.model
            if is_parallel(eval_model):
                eval_model = eval_model.module

        with adjust_status(eval_model, training=False):
            evaluator = StreamingEvaluator(
                model=eval_model,
                dataloader=self.get_eval_loader(),
                device=torch.device(self.device),
                num_classes=self.exp.num_classes
            )
            results = evaluator.evaluate()

        if self.rank == 0:
            self._show_evaluation_results(results=results)

        if torch.distributed.is_initialized():
            synchronize()

        self.save_ckpt("last_epoch", update_best_ckpt=True)
        if self.save_history_ckpt:
            self.save_ckpt(f"epoch_{self.epoch + 1}")

    def _show_evaluation_results(self, results: Dict) -> None:
        """Displays evaluation results."""
        print(f"Evaluation results at epoch {self.epoch + 1}:")
        for k, v in results.items():
            if isinstance(v, float):
                print(f"{k}: {v:.4f}")
            else:
                print(f"{k}:\n{v}")

        if self.args.logger == "tensorboard":
            for k, v in results.items():
                if isinstance(v, float):
                    self.tblogger.add_scalar(f"val/{k}", v, self.epoch + 1)