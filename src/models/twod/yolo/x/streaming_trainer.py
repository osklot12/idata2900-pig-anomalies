from typing import Dict

import torch

from yolox.core.trainer import Trainer
from yolox.data import DataPrefetcher
from yolox.utils import is_parallel, adjust_status, synchronize

from src.utils.logging import console

class StreamingTrainer(Trainer):
    """A custom trainer for streaming data."""

    def train_in_epoch(self):
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.train_loader = self.exp.get_data_loader(
                batch_size=self.args.batch_size,
                is_distributed=self.is_distributed,
                no_aug=self.no_aug,
                cache_img=self.args.cache,
            )
            self.prefetcher = DataPrefetcher(self.train_loader)
            self.max_iter = len(self.train_loader)

            self.before_epoch()
            self.train_in_iter()
            self.after_epoch()

    def get_train_loader(self):
        return self.train_loader

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
            console.log(f"[cyan]Evaluating...[/cyan]")

        if torch.distributed.is_initialized():
            synchronize()

        self.save_ckpt("last_epoch", update_best_ckpt=True)
        if self.save_history_ckpt:
            self.save_ckpt(f"epoch_{self.epoch + 1}")