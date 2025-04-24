import torch
from ultralytics import YOLO
from src.models.twod.yolo.viii.streaming_evaluator_viii import StreamingEvaluatorVIII


def main():
    # Load your trained YOLOv8 model (update this path!)
    model = YOLO('path/to/your/model.pt')  # e.g., 'runs/train/exp/weights/best.pt'
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # Dummy image: [B, C, H, W] = [1, 3, 640, 640]
    dummy_img = torch.randn(1, 3, 640, 640).to(device)

    print("✅ Wrapping model with StreamingEvaluatorVIII just for Concat debugging...")
    debug_wrapper = StreamingEvaluatorVIII(
        model=model.model,
        dataloader=None,
        device=device,
        num_classes=4  # ← use your actual number of classes
    )
    # Only calling _wrap_concat_debug(), not evaluate()

    print("🚀 Running one forward pass on dummy input")
    try:
        with torch.no_grad():
            model.model(dummy_img)  # Directly hit .model, bypass Ultralytics trainer
    except Exception as e:
        print("💥 Caught crash during inference:")
        print(e)


if __name__ == "__main__":
    main()
