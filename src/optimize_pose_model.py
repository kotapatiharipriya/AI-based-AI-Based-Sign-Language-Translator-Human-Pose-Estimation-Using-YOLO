
# src/optimize_pose_model.py

import argparse
import csv
import os
import time
from pathlib import Path

import cv2
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import yaml
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent.parent


# ----------------- Utility functions ----------------- #

def load_config(cfg_path: Path) -> dict:
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    # normalize paths
    cfg["base_model"] = str((ROOT / cfg["base_model"]).resolve())
    cfg["data_yaml"] = str((ROOT / cfg["data_yaml"]).resolve())
    cfg["val_images_dir"] = str((ROOT / cfg["val_images_dir"]).resolve())

    # device fallback if GPU not available
    device = str(cfg.get("device", "cpu"))
    if device != "cpu" and not torch.cuda.is_available():
        print("[WARN] CUDA not available, falling back to CPU.")
        device = "cpu"
    cfg["device"] = device

    return cfg


def compute_model_size_mb(path: Path) -> float:
    if not path.exists():
        return float("nan")
    size_bytes = os.path.getsize(path)
    return size_bytes / (1024 ** 2)


def compute_global_sparsity(model: nn.Module) -> float:
    total = 0
    zeros = 0
    with torch.no_grad():
        for p in model.parameters():
            t = p.numel()
            z = (p == 0).sum().item()
            total += t
            zeros += z
    return zeros / total if total > 0 else 0.0


def apply_global_pruning(yolo_model: YOLO, amount: float, module_types=None) -> float:
    """Apply global unstructured L1 pruning to Conv2d / Linear layers.

    Returns global sparsity ratio.
    """
    if module_types is None:
        module_types = ["Conv2d", "Linear"]

    module_types_set = set(module_types)

    modules_to_prune = []
    for m in yolo_model.model.modules():
        if m.__class__.__name__ in module_types_set:
            modules_to_prune.append(m)

    if not modules_to_prune:
        print("[WARN] No modules matched for pruning.")
        return 0.0

    parameters_to_prune = [(m, "weight") for m in modules_to_prune]

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )

    # remove reparam so weights are real tensors again
    for m, _ in parameters_to_prune:
        prune.remove(m, "weight")

    sparsity = compute_global_sparsity(yolo_model.model)
    return sparsity


def run_validation(yolo_model: YOLO, cfg: dict, half: bool = False) -> dict:
    """Run Ultralytics val() and extract a few pose metrics."""
    metrics = yolo_model.val(
        data=cfg["data_yaml"],
        imgsz=cfg["imgsz"],
        batch=cfg["batch"],
        device=cfg["device"],
        half=half,
        verbose=False,
    )

    # For pose models Ultralytics exposes metrics.pose.map, map50, map75 etc.
    pose_map = getattr(metrics.pose, "map", None)
    pose_map50 = getattr(metrics.pose, "map50", None)
    pose_map75 = getattr(metrics.pose, "map75", None)
    pose_p = None
    pose_r = None
    if hasattr(metrics.pose, "p") and len(metrics.pose.p) > 0:
        pose_p = metrics.pose.p[0]
    if hasattr(metrics.pose, "r") and len(metrics.pose.r) > 0:
        pose_r = metrics.pose.r[0]

    return {
        "pose_map": float(pose_map) if pose_map is not None else None,
        "pose_map50": float(pose_map50) if pose_map50 is not None else None,
        "pose_map75": float(pose_map75) if pose_map75 is not None else None,
        "pose_precision": float(pose_p) if pose_p is not None else None,
        "pose_recall": float(pose_r) if pose_r is not None else None,
    }


def benchmark_latency(yolo_model: YOLO, cfg: dict, half: bool = False) -> float:
    """Measure average per-image inference latency (ms)."""
    val_dir = Path(cfg["val_images_dir"])
    imgs = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        imgs.extend(sorted(val_dir.glob(ext)))

    if not imgs:
        print(f"[WARN] No images found in {val_dir}, skipping latency benchmark.")
        return float("nan")

    benchmark_cfg = cfg.get("benchmark", {})
    num_images = int(benchmark_cfg.get("num_images", 50))
    num_warmup = int(benchmark_cfg.get("num_warmup", 5))
    num_iters = int(benchmark_cfg.get("num_iters", 30))

    if num_images > 0:
        imgs = imgs[:num_images]

    device = cfg["device"]
    use_cuda = torch.cuda.is_available() and device != "cpu"

    def infer_on(img_path: Path):
        img = cv2.imread(str(img_path))
        if img is None:
            return
        _ = yolo_model.predict(
            source=img,
            imgsz=cfg["imgsz"],
            device=device,
            half=half,
            verbose=False,
        )

    # Warmup
    for i in range(num_warmup):
        infer_on(imgs[i % len(imgs)])

    # Timed runs
    times_ms = []
    for i in range(num_iters):
        img_path = imgs[i % len(imgs)]
        if use_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        infer_on(img_path)
        if use_cuda:
            torch.cuda.synchronize()
        dt_ms = (time.perf_counter() - t0) * 1000.0
        times_ms.append(dt_ms)

    return sum(times_ms) / len(times_ms)


# ----------------- Main experiment runner ----------------- #

def run_experiments(cfg: dict):
    base_model_path = Path(cfg["base_model"])
    device = cfg["device"]

    experiments = cfg.get("experiments", {})
    pruning_cfg = cfg.get("pruning", {})
    quant_cfg = cfg.get("quantization", {})
    bench_cfg = cfg.get("benchmark", {})
    results_csv = ROOT / bench_cfg.get("results_csv", "runs/pose/optimization_results.csv")
    results_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    print(f"Base model: {base_model_path}")
    print(f"Data YAML:  {cfg['data_yaml']}")
    print(f"Device:     {device}")

    # ---- 1. Baseline FP32 ----
    if experiments.get("baseline_fp32", True):
        print("\n========== Experiment: baseline_fp32 ==========")
        model_base = YOLO(str(base_model_path))
        model_base.to(device)

        val_metrics = run_validation(model_base, cfg, half=False)
        latency_ms = benchmark_latency(model_base, cfg, half=False)
        size_mb = compute_model_size_mb(base_model_path)

        rows.append({
            "variant": "baseline_fp32",
            "weights_path": str(base_model_path),
            "sparsity": compute_global_sparsity(model_base.model),
            "size_mb": size_mb,
            "latency_ms": latency_ms,
            **val_metrics,
        })

    # We'll build pruned model if requested
    pruned_path = base_model_path.with_name(base_model_path.stem + "_pruned.pt")

    if experiments.get("pruned_fp32", False) or experiments.get("pruned_fp16", False):
        if not pruning_cfg.get("enabled", False):
            print("\n[INFO] Pruned experiments requested but pruning.enabled=False, skipping pruning.")
        else:
            print("\n========== Pruning base model (unstructured L1) ==========")
            model_pruned = YOLO(str(base_model_path))
            model_pruned.to(device)
            model_pruned.model.eval()

            amount = float(pruning_cfg.get("global_amount", 0.3))
            module_types = pruning_cfg.get("module_types", ["Conv2d", "Linear"])
            print(f"Applying global unstructured pruning: amount={amount}, modules={module_types}")
            pruned_sparsity = apply_global_pruning(model_pruned, amount=amount, module_types=module_types)
            print(f"Global sparsity after pruning: {pruned_sparsity * 100:.2f}%")

            # Save pruned weights so we can reload later.
            model_pruned.save(str(pruned_path))
            print(f"Pruned model saved to: {pruned_path}")

    # ---- 2. Pruned FP32 ----
    if experiments.get("pruned_fp32", False) and pruning_cfg.get("enabled", False):
        print("\n========== Experiment: pruned_fp32 ==========")
        if not pruned_path.exists():
            print(f"[WARN] Expected pruned model at {pruned_path}, skipping pruned_fp32.")
        else:
            model_pruned = YOLO(str(pruned_path))
            model_pruned.to(device)

            val_metrics = run_validation(model_pruned, cfg, half=False)
            latency_ms = benchmark_latency(model_pruned, cfg, half=False)
            size_mb = compute_model_size_mb(pruned_path)
            sparsity_now = compute_global_sparsity(model_pruned.model)

            rows.append({
                "variant": "pruned_fp32",
                "weights_path": str(pruned_path),
                "sparsity": sparsity_now,
                "size_mb": size_mb,
                "latency_ms": latency_ms,
                **val_metrics,
            })

    # ---- 3. Pruned + FP16 (quantized inference) ----
    if (
        experiments.get("pruned_fp16", False)
        and pruning_cfg.get("enabled", False)
        and quant_cfg.get("enabled", False)
    ):
        if quant_cfg.get("type", "fp16").lower() != "fp16":
            print("\n[INFO] Only fp16 quantization is implemented in this script. Skipping pruned_fp16.")
        else:
            print("\n========== Experiment: pruned_fp16 (half precision inference) ==========")
            if not pruned_path.exists():
                print(f"[WARN] Expected pruned model at {pruned_path}, skipping pruned_fp16.")
            else:
                model_q = YOLO(str(pruned_path))
                model_q.to(device)

                # Do NOT call model_q.model.half() or model_q.save() here.
                # We just ask Ultralytics to run inference in FP16 via half=True.
                val_metrics = run_validation(model_q, cfg, half=True)
                latency_ms = benchmark_latency(model_q, cfg, half=True)

                # On disk we still use the same pruned FP32 weights; only inference runs in FP16.
                size_mb = compute_model_size_mb(pruned_path)
                sparsity_now = compute_global_sparsity(model_q.model)

                rows.append({
                    "variant": "pruned_fp16",
                    "weights_path": str(pruned_path),
                    "sparsity": sparsity_now,
                    "size_mb": size_mb,         # same file size as pruned_fp32
                    "latency_ms": latency_ms,
                    **val_metrics,
                })

    # ---- Save CSV + pretty print ----
    if rows:
        fieldnames = [
            "variant",
            "weights_path",
            "sparsity",
            "size_mb",
            "latency_ms",
            "pose_map",
            "pose_map50",
            "pose_map75",
            "pose_precision",
            "pose_recall",
        ]

        with open(results_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)

        print(f"\nResults saved to: {results_csv}\n")

        # Quick console summary
        print("Variant           | Sparsity | Size (MB) | Latency (ms/img) | pose mAP (0.5:0.95)")
        print("-" * 80)
        for r in rows:
            sparsity_pct = (
                r["sparsity"] * 100 if r["sparsity"] is not None else float("nan")
            )
            pose_map = r["pose_map"] if r["pose_map"] is not None else float("nan")
            print(
                f"{r['variant']:<16} | "
                f"{sparsity_pct:7.2f}% | "
                f"{r['size_mb']:8.2f} | "
                f"{r['latency_ms']:16.2f} | "
                f"{pose_map:15.4f}"
            )
    else:
        print("\n[INFO] No experiments were run (check config).")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/optimize_pose.yaml",
        help="Path to optimization config YAML",
    )
    args = parser.parse_args()

    cfg_path = (ROOT / args.config).resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    cfg = load_config(cfg_path)
    run_experiments(cfg)


if __name__ == "__main__":
    main()
