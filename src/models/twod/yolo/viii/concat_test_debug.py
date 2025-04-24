import torch
from ultralytics import YOLO
from src.models.twod.yolo.viii.streaming_evaluator_viii import StreamingEvaluatorVIII


def main():
    # Load YOLOv8 model
    model = YOLO('yolov8s.pt')
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # Move model to correct device
    model.model.to(device)

    # Dummy image
    dummy_img = torch.randn(1, 3, 640, 640).to(device)

    print("✅ Wrapping model with StreamingEvaluatorVIII just for Concat debugging...")
    debug_wrapper = StreamingEvaluatorVIII(
        model=model.model,
        dataloader=None,
        device=device,
        num_classes=4
    )

    print("🚀 Running one forward pass on dummy input")
    try:
        with torch.no_grad():
            model.model(dummy_img)
    except Exception as e:
        print("💥 Caught crash during inference:")
        print(e)

if __name__ == "__main__":
    main()
