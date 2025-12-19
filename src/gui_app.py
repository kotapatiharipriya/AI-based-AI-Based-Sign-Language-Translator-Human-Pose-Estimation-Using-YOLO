# src/gui_app.py

import argparse
import threading
from pathlib import Path
from tkinter import Tk, Button, Label, filedialog, StringVar, Frame

import cv2
from PIL import Image, ImageTk
from ultralytics import YOLO


ROOT_DIR = Path(__file__).resolve().parent.parent

DEFAULT_PRUNED_PATH = (
    ROOT_DIR / "runs" / "pose" / "pose-estimation-yolo" / "weights" / "best_pruned.pt"
)
DEFAULT_PRUNED_FP16_PATH = (
    ROOT_DIR
    / "runs"
    / "pose"
    / "pose-estimation-yolo"
    / "weights"
    / "best_pruned_fp16.pt"
)


class PoseCompareGUI:
    """
    GUI to compare pruned FP32 and pruned FP16 inference on the same image.

    Left panel  : best_pruned.pt (FP32)
    Right panel : best_pruned_fp16.pt (FP16) if file exists,
                  otherwise pruned model run with half=True.
    """

    def __init__(self, root, model_pruned_path: Path, model_pruned_fp16_path: Path):
        self.root = root
        self.root.title("YOLO Pose - Pruned vs Pruned FP16 Comparison")

        self.model_pruned_path = model_pruned_path
        self.model_pruned_fp16_path = model_pruned_fp16_path

        # Status and info labels
        self.status_var = StringVar()
        self.status_var.set("Select an image to start comparison...")

        # Load pruned FP32 model
        if not self.model_pruned_path.exists():
            raise FileNotFoundError(f"Pruned model not found: {self.model_pruned_path}")
        self.model_pruned = YOLO(str(self.model_pruned_path))

        # Try to load FP16 weights; if not available, we'll use pruned model with half=True
        self.has_separate_fp16_weights = self.model_pruned_fp16_path.exists()
        if self.has_separate_fp16_weights:
            self.model_pruned_fp16 = YOLO(str(self.model_pruned_fp16_path))
            fp16_info = self.model_pruned_fp16_path.name
        else:
            self.model_pruned_fp16 = self.model_pruned  # reuse pruned model
            fp16_info = "using pruned model with half=True (no separate *_fp16.pt file)"

        info_text = (
            f"Pruned FP32 model: {self.model_pruned_path.name}\n"
            f"Pruned FP16 model: {fp16_info}\n"
            "Load an image, then click the buttons to run each model."
        )

        self.info_label = Label(root, text=info_text, justify="left")
        self.info_label.pack(pady=5)

        self.status_label = Label(root, textvariable=self.status_var)
        self.status_label.pack(pady=5)

        # Buttons frame
        btn_frame = Frame(root)
        btn_frame.pack(pady=5)

        self.open_button = Button(btn_frame, text="Open Image", command=self.open_image)
        self.open_button.grid(row=0, column=0, padx=5)

        self.run_fp32_button = Button(
            btn_frame, text="Run Pruned FP32", command=lambda: self.start_inference("fp32")
        )
        self.run_fp32_button.grid(row=0, column=1, padx=5)

        self.run_fp16_button = Button(
            btn_frame, text="Run Pruned FP16", command=lambda: self.start_inference("fp16")
        )
        self.run_fp16_button.grid(row=0, column=2, padx=5)

        # Two image panels side-by-side
        images_frame = Frame(root)
        images_frame.pack(padx=10, pady=10)

        self.label_fp32_title = Label(images_frame, text="Pruned FP32 (best_pruned.pt)")
        self.label_fp32_title.grid(row=0, column=0, pady=(0, 4))

        self.label_fp16_title = Label(
            images_frame,
            text="Pruned FP16 (best_pruned_fp16.pt or half=True)",
        )
        self.label_fp16_title.grid(row=0, column=1, pady=(0, 4))

        self.image_label_fp32 = Label(images_frame)
        self.image_label_fp32.grid(row=1, column=0, padx=5)

        self.image_label_fp16 = Label(images_frame)
        self.image_label_fp16.grid(row=1, column=1, padx=5)

        # Store current image path and PhotoImage objects to avoid GC
        self.current_image_path = None
        self._photo_fp32 = None
        self._photo_fp16 = None

    def open_image(self):
        filetypes = [
            ("Image files", "*.jpg *.jpeg *.png *.bmp"),
            ("All files", "*.*"),
        ]
        filename = filedialog.askopenfilename(title="Select image", filetypes=filetypes)
        if not filename:
            return

        self.current_image_path = filename
        self.status_var.set(f"Loaded image: {Path(filename).name}\nClick a button to run inference.")

        # Clear old outputs
        self.image_label_fp32.config(image="")
        self.image_label_fp16.config(image="")
        self._photo_fp32 = None
        self._photo_fp16 = None

    def start_inference(self, mode: str):
        if self.current_image_path is None:
            self.status_var.set("Please open an image first.")
            return

        if mode == "fp32":
            self.status_var.set("Running pruned FP32 inference...")
        elif mode == "fp16":
            self.status_var.set("Running pruned FP16 inference...")
        else:
            return

        thread = threading.Thread(target=self.run_inference, args=(mode,))
        thread.daemon = True
        thread.start()

    def run_inference(self, mode: str):
        try:
            img_path = self.current_image_path
            img = cv2.imread(img_path)
            if img is None:
                self.root.after(0, lambda: self.status_var.set(f"Failed to read image: {img_path}"))
                return

            max_size = 800

            if mode == "fp32":
                results = self.model_pruned.predict(
                    source=img, imgsz=640, conf=0.25, verbose=False, half=False
                )
                result = results[0]
                annotated_bgr = result.plot()
                annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

                h, w = annotated_rgb.shape[:2]
                scale = min(max_size / max(h, w), 1.0)
                if scale < 1.0:
                    annotated_rgb = cv2.resize(
                        annotated_rgb,
                        (int(w * scale), int(h * scale)),
                        interpolation=cv2.INTER_AREA,
                    )

                img_pil = Image.fromarray(annotated_rgb)
                self._photo_fp32 = ImageTk.PhotoImage(img_pil)
                self.root.after(0, lambda: self.image_label_fp32.config(image=self._photo_fp32))
                self.root.after(
                    0,
                    lambda: self.status_var.set(
                        f"Done! Pruned FP32 inference for {Path(img_path).name}"
                    ),
                )

            elif mode == "fp16":
                # Use separate FP16 weights if available, otherwise same pruned model with half=True
                model = self.model_pruned_fp16
                results = model.predict(
                    source=img,
                    imgsz=640,
                    conf=0.25,
                    verbose=False,
                    half=True,  # run in half precision on device (GPU)
                )
                result = results[0]
                annotated_bgr = result.plot()
                annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

                h, w = annotated_rgb.shape[:2]
                scale = min(max_size / max(h, w), 1.0)
                if scale < 1.0:
                    annotated_rgb = cv2.resize(
                        annotated_rgb,
                        (int(w * scale), int(h * scale)),
                        interpolation=cv2.INTER_AREA,
                    )

                img_pil = Image.fromarray(annotated_rgb)
                self._photo_fp16 = ImageTk.PhotoImage(img_pil)
                self.root.after(0, lambda: self.image_label_fp16.config(image=self._photo_fp16))
                self.root.after(
                    0,
                    lambda: self.status_var.set(
                        f"Done! Pruned FP16 inference for {Path(img_path).name}"
                    ),
                )

        except Exception as e:
            self.root.after(0, lambda: self.status_var.set(f"Error during inference: {e}"))


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO Pose GUI - Pruned vs Pruned FP16")
    parser.add_argument(
        "--model_pruned",
        type=str,
        default=str(DEFAULT_PRUNED_PATH),
        help="Path to pruned FP32 weights (e.g., best_pruned.pt)",
    )
    parser.add_argument(
        "--model_pruned_fp16",
        type=str,
        default=str(DEFAULT_PRUNED_FP16_PATH),
        help=(
            "Path to pruned FP16 weights (e.g., best_pruned_fp16.pt). "
            "If not found, GUI will reuse pruned FP32 weights and run them with half=True."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model_pruned_path = Path(args.model_pruned)
    model_pruned_fp16_path = Path(args.model_pruned_fp16)

    root = Tk()
    app = PoseCompareGUI(root, model_pruned_path, model_pruned_fp16_path)
    root.mainloop()


if __name__ == "__main__":
    main()
