
# src/infer_single_image.py

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


def predict_and_save(model_path, image_path, output_path):
    model = YOLO(model_path)
    results = model.predict(source=image_path, imgsz=640, conf=0.25, verbose=False)

    # We expect a single image -> results[0]
    result = results[0]
    plotted = result.plot()  # BGR ndarray with keypoints drawn

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), plotted)
    print(f"Saved annotated image to: {output_path}")

    # Also print keypoints for inspection
    if result.keypoints is not None:
        kpts = result.keypoints.xy[0].cpu().numpy()
        print("Keypoints (x,y) in pixels:")
        print(kpts)


def main():
    parser = argparse.ArgumentParser(description="Run pose inference on a single image")
    parser.add_argument(
        "--model",
        type=str,
        default="runs/pose/pose-estimation-yolo/weights/best.pt",
        help="Path to YOLO pose model weights (.pt)",
    )
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument(
        "--out", type=str, default="outputs/annotated.png", help="Path to save annotated image"
    )
    args = parser.parse_args()

    predict_and_save(args.model, args.image, args.out)


if __name__ == "__main__":
    main()
