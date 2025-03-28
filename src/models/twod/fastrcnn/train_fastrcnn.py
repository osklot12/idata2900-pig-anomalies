import os
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from src.models.model_trainer import ModelTrainer
from src.network.client.client_network import NetworkClient
from src.network.client.client_context import ClientContext
from src.network.messages.requests.get_frame_batch_request import GetFrameBatchRequest
from src.data.dataset_split import DatasetSplit

class RCNNTrainer(ModelTrainer):
    def __init__(self, client: NetworkClient, context: ClientContext):
        self.client = client
        self.context = context
        self.model = None

    def setup(self):
        self.model = self.create_model()

    def create_model(self):
        model = fasterrcnn_resnet50_fpn(weights=None, num_classes=2)
        return model.cuda()

    def train(self) -> str:
        self.client.connect()
        self.setup()

        model = self.model
        model.train()

        optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
        save_dir = "checkpoints"
        os.makedirs(save_dir, exist_ok=True)

        try:
            for iteration in range(1000):
                request = GetFrameBatchRequest(DatasetSplit.TRAIN, batch_size=8)
                response = self.client.send_request(request)
                batch = response.execute(self.context)

                images, targets = self._convert_to_tensors(batch)

                optimizer.zero_grad()
                loss_dict = model(images, targets)
                losses = sum(loss_dict.values())
                total_loss = losses if isinstance(losses, torch.Tensor) else torch.tensor(losses, dtype=torch.float32, device="cuda")
                total_loss.backward()
                optimizer.step()

                if iteration % 100 == 0:
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