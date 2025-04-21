import tempfile
from types import SimpleNamespace

import yaml
import os
from datetime import datetime
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from ultralytics.models.yolo.obb import OBBTrainer
import torch


class ResettableDataLoader:
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.dataset = dataloader.dataset
        self.num_workers = dataloader.num_workers

    def __iter__(self):
        return iter(self.dataloader)

    def __len__(self):
        return len(self.dataloader)

    def reset(self):
        # No-op, just to satisfy Ultralytics' assumption
        pass


class TrainingSetup:
    def __init__(self, dataset, eval_dataset=None, model_path="yolo11m-obb.pt", log_dir="runs", epochs=300, imgsz=640):
        self.metrics = None
        self.dataset = dataset
        self.model_path = model_path
        self.epochs = epochs
        self.imgsz = imgsz
        self.eval_dataset = eval_dataset or dataset

        # TensorBoard logging
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.log_dir = os.path.join(log_dir, f"yolo11m_obb_{timestamp}")
        self.writer = SummaryWriter(log_dir=self.log_dir)

    def _log_epoch_metrics(self, trainer):
        epoch = trainer.epoch
        metrics = getattr(trainer, "metrics", {})
        print(f"📊 Epoch {epoch} metrics: {metrics}")
        if not metrics:
            return
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f"metrics/{key}", value, epoch)


    def train(self):
        dummy_root = tempfile.mkdtemp(prefix="yolo_dummy_")
        os.makedirs(os.path.join(dummy_root, "train", "images"), exist_ok=True)
        os.makedirs(os.path.join(dummy_root, "val", "images"), exist_ok=True)

        dummy_yaml = {
            "train": os.path.join(dummy_root, "train"),
            "val": os.path.join(dummy_root, "val"),
            "nc": 4,
            "names": ["tail-biting", "ear-biting", "belly-nosing", "tail-down"]
        }

        dummy_yaml_path = os.path.join(dummy_root, "data.yaml")
        with open(dummy_yaml_path, "w") as f:
            yaml.safe_dump(dummy_yaml, f)

        overrides = {
            "model": self.model_path,
            "data": dummy_yaml_path,
            "epochs": self.epochs,
            "imgsz": self.imgsz,
            "project": self.log_dir,
            "name": "train",
            "save": True,
            "verbose": True,
            "val": True,
            "device": 0 if torch.cuda.is_available() else "cpu",
        }

        trainer = OBBTrainer(overrides=overrides)

        def variable_length_collate(batch):
            return {
                "img": torch.stack([item["img"] for item in batch], dim=0),
                "cls": torch.cat([item["instances"]["cls"] for item in batch], dim=0),
                "bboxes": torch.cat([item["instances"]["bboxes"] for item in batch], dim=0),
                "batch_idx": torch.cat([item["batch_idx"] for item in batch], dim=0),
                "ori_shape": [item["ori_shape"][0] for item in batch],
                "ratio_pad": [item["ratio_pad"][0] for item in batch],
                "im_file": [item["im_file"][0] for item in batch],
            }

        def patched_get_dataloader(_, dataset_path, batch_size, rank=0, mode="train"):
            dataset = self.dataset if mode == "train" else self.eval_dataset
            print(f"📥 Creating DataLoader for mode={mode}, batch_size={batch_size}")
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                drop_last=False,
                collate_fn=variable_length_collate
            )
            return ResettableDataLoader(loader)

        trainer.get_dataloader = patched_get_dataloader.__get__(trainer)

        def skip_plot_labels(self):
            print("⚠️ Skipping plot_training_labels() — in-memory dataset has no .labels.")

        trainer.plot_training_labels = skip_plot_labels.__get__(trainer)

        def patched_plot_training_samples(self, batch, ni):
            print("⚠️ Skipping plot_training_samples() — missing im_file in in-memory batches.")
            im = batch["img"]
            cls = batch["cls"].view(-1, 1).float()
            bboxes = batch["bboxes"].float()
            targets = torch.cat([cls, bboxes], dim=1)
            return im, targets

        trainer.plot_training_samples = patched_plot_training_samples.__get__(trainer)

        trainer.add_callback("on_fit_epoch_end", self._log_epoch_metrics)

        try:
            trainer.train()
        except Exception as e:
            print("\n💥 Training crashed with an exception:")
            import traceback
            traceback.print_exc()

        # ✅ Defensive assignment
        final_metrics = getattr(trainer, "metrics", {})
        if final_metrics is None:
            final_metrics = {}

        self.metrics = final_metrics

        for key, value in final_metrics.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f"final_metrics/{key}", value, self.epochs)

        self.writer.flush()
        self.writer.close()



