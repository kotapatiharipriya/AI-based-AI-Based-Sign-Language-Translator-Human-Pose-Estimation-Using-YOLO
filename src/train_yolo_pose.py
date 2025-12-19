
# src/train_yolo_pose.py

from pathlib import Path

from ultralytics import YOLO


def main():
    root = Path(__file__).resolve().parent.parent

    data_yaml = root / "configs" / "pose_estimation.yaml"

    # You can switch to another pose model if desired, e.g. 'yolo11n-pose.pt'
    pretrained_weights = "yolov8n-pose.pt"

    model = YOLO(pretrained_weights)  # load pretrained pose model

    results = model.train(
        data=str(data_yaml),
        task="pose",
        epochs=50,
        imgsz=640,
        batch=4,
        name="pose-estimation-yolo",
        project=str(root / "runs" / "pose"),
        lr0=1e-3,
        patience=10,
        verbose=True,
    )

    print("Training done. Best model directory:")
    print(results.save_dir)
    print("Weights are typically under 'weights/best.pt' inside that directory.")


if __name__ == "__main__":
    main()
