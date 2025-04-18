from typing import Dict

import torch

from src.models.twod.yolo.x.streaming_evaluator import StreamingEvaluator
from yolox.core.trainer import Trainer
from yolox.utils import is_parallel, adjust_status, synchronize


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
            try:
                results = evaluator.evaluate()
            except Exception as e:
                print(f"[Trainer] ❌ Exception during evaluation: {e}")

        print(f"Finished evaluation")
        if self.rank == 0:
            print(f"Showing results")
            self._show_evaluation_results(results=results)

        print(f"Finished showing evaluation results")

        if torch.distributed.is_initialized():
            print(f"[StreamingTrainer] IT IS DISTRIBUTED!")
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