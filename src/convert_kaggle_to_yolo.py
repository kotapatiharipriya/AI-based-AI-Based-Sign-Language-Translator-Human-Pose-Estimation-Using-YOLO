# src/convert_kaggle_to_yolo.py

import os
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
from tqdm import tqdm

# -------- CONFIG --------
ROOT_DIR = Path(__file__).resolve().parent.parent
KAGGLE_ROOT = ROOT_DIR / "data" / "kaggle_raw"
XML_PATH = KAGGLE_ROOT / "annotations.xml"
IMAGES_DIR = KAGGLE_ROOT / "images"

YOLO_ROOT = ROOT_DIR / "data" / "yolo_format"
TRAIN_SPLIT = 0.8
RANDOM_SEED = 42

# COCO 17 keypoints order (Ultralytics default):
# 0 nose
# 1 left_eye
# 2 right_eye
# 3 left_ear
# 4 right_ear
# 5 left_shoulder
# 6 right_shoulder
# 7 left_elbow
# 8 right_elbow
# 9 left_wrist
# 10 right_wrist
# 11 left_hip
# 12 right_hip
# 13 left_knee
# 14 right_knee
# 15 left_ankle
# 16 right_ankle
#
# Dataset (18 keypoints):
# 0 Nose
# 1 Neck
# 2 Right shoulder
# 3 Right elbow
# 4 Right wrist
# 5 Left shoulder
# 6 Left elbow
# 7 Left wrist
# 8 Right hip
# 9 Right knee
# 10 Right foot
# 11 Left hip
# 12 Left knee
# 13 Left foot
# 14 Right eye
# 15 Left eye
# 16 Right ear
# 17 Left ear
#
# Map dataset indices -> COCO indices (we drop Neck (1)):
DATASET_TO_COCO = [
    0,   # 0 nose          <- dataset 0
    15,  # 1 left_eye      <- dataset 15
    14,  # 2 right_eye     <- dataset 14
    17,  # 3 left_ear      <- dataset 17
    16,  # 4 right_ear     <- dataset 16
    5,   # 5 left_shoulder <- dataset 5
    2,   # 6 right_shoulder<- dataset 2
    6,   # 7 left_elbow    <- dataset 6
    3,   # 8 right_elbow   <- dataset 3
    7,   # 9 left_wrist    <- dataset 7
    4,   # 10 right_wrist  <- dataset 4
    11,  # 11 left_hip     <- dataset 11
    8,   # 12 right_hip    <- dataset 8
    12,  # 13 left_knee    <- dataset 12
    9,   # 14 right_knee   <- dataset 9
    13,  # 15 left_ankle   <- dataset 13 (left foot)
    10,  # 16 right_ankle  <- dataset 10 (right foot)
]


def parse_image_element(img_el):
    """
    Parse a single <image> element from annotations.xml.

    Returns:
        image_name (str)
        width (int), height (int)
        bbox: (x1, y1, x2, y2) or None
        kpts: {dataset_index: (x, y, v)}
    """
    image_name = img_el.get("name")
    width = int(float(img_el.get("width", 0)))
    height = int(float(img_el.get("height", 0)))

    # 1) Bounding box (if present)
    bbox = None
    for box in img_el.findall("box"):
        label = box.get("label", "")
        if label.lower() in ("human_box", "person", "human"):
            x1 = float(box.get("xtl"))
            y1 = float(box.get("ytl"))
            x2 = float(box.get("xbr"))
            y2 = float(box.get("ybr"))
            bbox = (x1, y1, x2, y2)
            break  # assume single person

    # 2) Keypoints
    kpts = {}
    for p in img_el.findall("points"):
        label = p.get("label", None)
        if label is None:
            continue
        try:
            idx = int(label)
        except ValueError:
            continue  # not a numeric label

        pts_str = p.get("points", "")
        if not pts_str:
            continue
        try:
            x_str, y_str = pts_str.split(",")
        except ValueError:
            continue

        x = float(x_str)
        y = float(y_str)

        # Visibility from attributes: Presumed_Location or Russian equivalent
        vis_raw = None
        # CVAT style: <attributes><attribute name="Presumed_Location">true/false</attribute></attributes>
        for attr in p.findall(".//attribute"):
            name = attr.get("name", "")
            if name in ("Presumed_Location", "Предположительное_значение"):
                vis_raw = attr.text
                break

        is_presumed = str(vis_raw).lower() == "true"
        # YOLO visibility: 0=not labeled, 1=labeled but not visible, 2=visible
        v = 1 if is_presumed else 2
        kpts[idx] = (x, y, v)

    return image_name, width, height, bbox, kpts


def remap_and_normalize_keypoints(kpts_old, img_w, img_h):
    """
    Convert from dataset's 18-keypoint scheme to COCO's 17-keypoint scheme,
    and normalize x,y to [0,1]. Missing -> (0,0,0).
    Returns [x1, y1, v1, ..., x17, y17, v17].
    """
    kpts_new = []
    for dataset_idx in DATASET_TO_COCO:
        if dataset_idx in kpts_old:
            x, y, v = kpts_old[dataset_idx]
            x_n = x / img_w
            y_n = y / img_h
        else:
            x_n, y_n, v = 0.0, 0.0, 0
        kpts_new.extend([x_n, y_n, float(v)])
    return kpts_new


def bbox_to_yolo(bbox, img_w, img_h, kpts_old):
    """
    Convert bbox to YOLO normalized format (x_c, y_c, w, h).
    If bbox is None, compute from keypoints.
    """
    if bbox is not None:
        x1, y1, x2, y2 = bbox
    else:
        xs = [x for (_, (x, y, v)) in kpts_old.items() if v > 0]
        ys = [y for (_, (x, y, v)) in kpts_old.items() if v > 0]
        if not xs or not ys:
            return None
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)

    w = x2 - x1
    h = y2 - y1
    x_c = x1 + w / 2.0
    y_c = y1 + h / 2.0

    return x_c / img_w, y_c / img_h, w / img_w, h / img_h


def main():
    assert XML_PATH.exists(), f"annotations.xml not found at {XML_PATH}"
    assert IMAGES_DIR.exists(), f"Images folder not found at {IMAGES_DIR}"

    tree = ET.parse(XML_PATH)
    root = tree.getroot()

    # Collect all <image> elements
    image_elements = list(root.findall(".//image"))
    if not image_elements:
        raise RuntimeError("No <image> elements found in annotations.xml")

    # Prepare output dirs
    (YOLO_ROOT / "images" / "train").mkdir(parents=True, exist_ok=True)
    (YOLO_ROOT / "images" / "val").mkdir(parents=True, exist_ok=True)
    (YOLO_ROOT / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (YOLO_ROOT / "labels" / "val").mkdir(parents=True, exist_ok=True)

    indices = list(range(len(image_elements)))
    random.seed(RANDOM_SEED)
    random.shuffle(indices)

    n_train = int(len(indices) * TRAIN_SPLIT)
    train_idx_set = set(indices[:n_train])

    for i, img_el in tqdm(
        list(enumerate(image_elements)),
        total=len(image_elements),
        desc="Converting XML to YOLO format",
    ):
        image_name, img_w, img_h, bbox, kpts_old = parse_image_element(img_el)

        if not image_name:
            print("[WARN] <image> without name, skipping")
            continue

        src_img_path = IMAGES_DIR / image_name
        if not src_img_path.exists():
            print(f"[WARN] Image not found: {src_img_path}, skipping")
            continue

        if not kpts_old:
            print(f"[WARN] No keypoints for {image_name}, skipping")
            continue

        box_yolo = bbox_to_yolo(bbox, img_w, img_h, kpts_old)
        if box_yolo is None:
            print(f"[WARN] Could not compute bbox for {image_name}, skipping")
            continue

        x_c, y_c, w, h = box_yolo
        kpts = remap_and_normalize_keypoints(kpts_old, img_w, img_h)

        subset = "train" if i in train_idx_set else "val"

        dst_img_path = YOLO_ROOT / "images" / subset / image_name
        dst_label_path = YOLO_ROOT / "labels" / subset / (Path(image_name).stem + ".txt")

        print("SRC:", src_img_path)
        print("DST:", dst_img_path)

        shutil.copy2(src_img_path, dst_img_path)

        # YOLO pose label: class x y w h kpt1x kpt1y v1 ...
        with open(dst_label_path, "w") as f:
            line = "0 "  # class id = 0 ("person")
            line += f"{x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f} "
            line += " ".join(f"{val:.6f}" for val in kpts)
            f.write(line.strip() + "\n")

    print("Conversion complete!")
    print(f"YOLO dataset created under: {YOLO_ROOT}")


if __name__ == "__main__":
    main()

