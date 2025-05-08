import os

import torch
from torch.distributed._composable.replicate import DDP
from torch.utils.tensorboard import SummaryWriter

from yolox.core.trainer import Trainer
from yolox.data import DataPrefetcher
from yolox.utils import is_parallel, adjust_status, synchronize, get_model_info, occupy_mem, ModelEMA, \
    WandbLogger, MlflowLogger
from loguru import logger

from src.models.twod.yolo.x.yolox_predictor import YOLOXPredictor

from src.utils.logging import console

class StreamingTrainer(Trainer):
    """A custom trainer for streaming data."""

    def before_train(self):
        logger.info("args: {}".format(self.args))
        logger.info("exp value:\n{}".format(self.exp))

        # model related init
        torch.cuda.set_device(self.local_rank)
        model = self.exp.get_model()

        if self.exp.freeze_backbone:
                self._freeze_yolox_backbone(model, freeze_neck=True)
                console.log("[cyan]Backbone and neck frozen. Training head only.[/cyan]")

        logger.info(
            "Model Summary: {}".format(get_model_info(model, self.exp.test_size))
        )
        model.to(self.device)

        # solver related init
        self.optimizer = self.exp.get_optimizer(self.args.batch_size)

        # value of epoch will be set in `resume_train`
        model = self.resume_train(model)

        # data related init
        self.no_aug = self.start_epoch >= self.max_epoch - self.exp.no_aug_epochs
        self.train_loader = self.exp.get_data_loader(
            batch_size=self.args.batch_size,
            is_distributed=self.is_distributed,
            no_aug=self.no_aug,
            cache_img=self.args.cache,
        )
        logger.info("init prefetcher, this might take one minute or less...")
        self.prefetcher = DataPrefetcher(self.train_loader)
        # max_iter means iters per epoch
        self.max_iter = len(self.train_loader)

        self.lr_scheduler = self.exp.get_lr_scheduler(
            self.exp.basic_lr_per_img * self.args.batch_size, self.max_iter
        )
        if self.args.occupy:
            occupy_mem(self.local_rank)

        if self.is_distributed:
            model = DDP(model, device_ids=[self.local_rank], broadcast_buffers=False)

        if self.use_model_ema:
            self.ema_model = ModelEMA(model, 0.9998)
            self.ema_model.updates = self.max_iter * self.start_epoch

        self.model = model

        self.evaluator = self.exp.get_evaluator(
            batch_size=self.args.batch_size, is_distributed=self.is_distributed
        )
        # Tensorboard and Wandb loggers
        if self.rank == 0:
            if self.args.logger == "tensorboard":
                self.tblogger = SummaryWriter(os.path.join(self.file_name, "tensorboard"))
            elif self.args.logger == "wandb":
                self.wandb_logger = WandbLogger.initialize_wandb_logger(
                    self.args,
                    self.exp,
                    self.evaluator.dataloader.dataset
                )
            elif self.args.logger == "mlflow":
                self.mlflow_logger = MlflowLogger()
                self.mlflow_logger.setup(args=self.args, exp=self.exp)
            else:
                raise ValueError("logger must be either 'tensorboard', 'mlflow' or 'wandb'")

        logger.info("Training start...")
        logger.info("\n{}".format(model))

    @staticmethod
    def _freeze_yolox_backbone(model: torch.nn.Module, freeze_neck: bool = True):
        for name, param in model.named_parameters():
            if "backbone" in name or (freeze_neck and "neck" in name):
                param.requires_grad = False
                console.log(f"[dim]Froze: {name}[/dim]")

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
            predictor = YOLOXPredictor(
                model=eval_model,
                device=torch.device(self.device),
                conf_thresh=0.5
            )
            self.exp.evaluator.evaluate(predictor)

        if torch.distributed.is_initialized():
            synchronize()

        self.save_ckpt("last_epoch", update_best_ckpt=True)
        if self.save_history_ckpt:
            self.save_ckpt(f"epoch_{self.epoch + 1}")