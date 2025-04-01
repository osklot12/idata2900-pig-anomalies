import os
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from src.models.model_trainer import ModelTrainer
from src.network.client.network_client import NetworkClient
from src.network.messages.requests.get_frame_batch_request import GetFrameBatchRequest
from src.data.dataset.dataset_split import DatasetSplit
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RCNNTrainer(ModelTrainer):
    def __init__(self, client: NetworkClient):
        self.client = client
        self.model = None

    def setup(self):
        self.model = self.create_model()

    def create_model(self):
        model = fasterrcnn_resnet50_fpn(weights=None, num_classes=2)
        return model.cuda()

    def train(self) -> str:
        self.client.connect()
        logger.info("Connected to server.")

        self.setup()

        model = self.model
        model.train()

        optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
        save_dir = "checkpoints"
        os.makedirs(save_dir, exist_ok=True)

        loss_accumulator = []

        try:
            for iteration in range(1000):
                request = GetFrameBatchRequest(DatasetSplit.TRAIN, batch_size=8)
                response = self.client.send_request(request)
                batch = response.execute()

                images, targets = self._convert_to_tensors(batch)

                optimizer.zero_grad()
                loss_dict = model(images, targets)
                losses = sum(loss_dict.values())
                total_loss = losses if isinstance(losses, torch.Tensor) else torch.tensor(losses, dtype=torch.float32, device="cuda")
                total_loss.backward()
                optimizer.step()

                # Log each individual loss component
                loss_strings = [f"{k}: {v.item():.4f}" for k, v in loss_dict.items()]
                logger.info(f"Iteration {iteration} losses: {', '.join(loss_strings)}")

                loss_accumulator.append(total_loss.item())

                if iteration % 100 == 0 and iteration > 0:
                    avg_loss = sum(loss_accumulator[-100:]) / 100
                    logger.info(f"Iteration {iteration}: Average loss over last 100 iters: {avg_loss:.4f}")

                    checkpoint_path = os.path.join(save_dir, f"model_checkpoint_{iteration}.pt")
                    torch.save(model.state_dict(), checkpoint_path)

            final_path = os.path.join(save_dir, "model_final.pt")
            torch.save(model.state_dict(), final_path)

            return "Training completed."

        finally:
            self.client.close()

    def evaluate(self) -> str:
        return "Evaluation not implemented yet."

    def _convert_to_tensors(self, batch):
        images = []
        targets = []
        for frame in batch:
            images.append(torch.tensor(frame.image, dtype=torch.float32, device="cuda").permute(2, 0, 1) / 255.0)
            boxes = torch.tensor([
                [ann.bbox[0], ann.bbox[1], ann.bbox[0] + ann.bbox[2], ann.bbox[1] + ann.bbox[3]]
                for ann in frame.annotations
            ], dtype=torch.float32, device="cuda")
            labels = torch.tensor([ann.category_id for ann in frame.annotations], dtype=torch.int64, device="cuda")
            targets.append({"boxes": boxes, "labels": labels})
        return images, targets
