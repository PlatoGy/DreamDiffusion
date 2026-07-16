import argparse
import csv
import json
import math
import re
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


TEST_IMAGE_RE = re.compile(r"^test(?P<sample>\d+)-(?P<copy>\d+)\.png$")


def load_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)


def safe_pcc(gt, pred):
    gt_flat = gt.reshape(-1)
    pred_flat = pred.reshape(-1)
    if np.std(gt_flat) == 0 or np.std(pred_flat) == 0:
        return float("nan")
    return float(np.corrcoef(gt_flat, pred_flat)[0, 1])


def safe_cosine(gt, pred):
    gt_flat = gt.reshape(-1)
    pred_flat = pred.reshape(-1)
    denom = np.linalg.norm(gt_flat) * np.linalg.norm(pred_flat)
    if denom == 0:
        return float("nan")
    return float(np.dot(gt_flat, pred_flat) / denom)


def compute_basic_metrics(gt, pred):
    diff = pred - gt
    mse = float(np.mean(diff ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(diff)))
    pcc = safe_pcc(gt, pred)
    ssim = float(structural_similarity(gt, pred, data_range=255, channel_axis=-1))
    psnr = float(peak_signal_noise_ratio(gt, pred, data_range=255))
    cosine = safe_cosine(gt, pred)
    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "pcc": pcc,
        "ssim": ssim,
        "psnr": psnr,
        "cosine": cosine,
    }


def make_lpips_metric(device):
    import torch
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    metric = LearnedPerceptualImagePatchSimilarity(net_type="alex").to(device)
    metric.eval()

    def lpips(gt, pred):
        gt_t = torch.from_numpy(gt).permute(2, 0, 1).unsqueeze(0).to(device)
        pred_t = torch.from_numpy(pred).permute(2, 0, 1).unsqueeze(0).to(device)
        gt_t = gt_t / 127.5 - 1.0
        pred_t = pred_t / 127.5 - 1.0
        with torch.no_grad():
            value = metric(pred_t, gt_t).item()
        return float(value)

    return lpips


def collect_pairs(run_dir, gt_index=0, prediction_index=None):
    grouped = {}
    for path in run_dir.glob("test*-*.png"):
        match = TEST_IMAGE_RE.match(path.name)
        if match is None:
            continue
        sample_id = int(match.group("sample"))
        copy_id = int(match.group("copy"))
        grouped.setdefault(sample_id, {})[copy_id] = path

    rows = []
    for sample_id in sorted(grouped):
        copies = grouped[sample_id]
        gt_path = copies.get(gt_index)
        if gt_path is None:
            continue
        pred_ids = [prediction_index] if prediction_index is not None else sorted(
            copy_id for copy_id in copies if copy_id != gt_index
        )
        for pred_id in pred_ids:
            pred_path = copies.get(pred_id)
            if pred_path is None:
                continue
            rows.append(
                {
                    "sample_id": sample_id,
                    "ground_truth": gt_path,
                    "prediction_id": pred_id,
                    "prediction": pred_path,
                }
            )
    return rows


def nanmean(values):
    values = [v for v in values if not (isinstance(v, float) and math.isnan(v))]
    if not values:
        return float("nan")
    return float(np.mean(values))


def summarize(rows, metric_names):
    pair_mean = {name: nanmean([row[name] for row in rows]) for name in metric_names}

    sample_ids = sorted({row["sample_id"] for row in rows})
    sample_rows = []
    for sample_id in sample_ids:
        sample_group = [row for row in rows if row["sample_id"] == sample_id]
        sample_summary = {"sample_id": sample_id, "num_predictions": len(sample_group)}
        for name in metric_names:
            sample_summary[name] = nanmean([row[name] for row in sample_group])
        sample_rows.append(sample_summary)

    sample_mean = {name: nanmean([row[name] for row in sample_rows]) for name in metric_names}
    return pair_mean, sample_rows, sample_mean


def write_csv(path, rows, fieldnames):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def json_ready(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate DreamDiffusion images saved by gen_eval_eeg.py."
    )
    parser.add_argument("run_dir", type=Path, help="A results/eval/<timestamp> directory.")
    parser.add_argument("--gt-index", type=int, default=0, help="Ground-truth image copy index. Default: 0.")
    parser.add_argument(
        "--prediction-index",
        type=int,
        default=None,
        help="Evaluate only one generated copy index, e.g. 1. Default: evaluate all generated copies.",
    )
    parser.add_argument("--output-prefix", default="generated_eval", help="Output file prefix.")
    parser.add_argument(
        "--lpips",
        action="store_true",
        help="Also compute LPIPS. This may download AlexNet weights the first time.",
    )
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Device for LPIPS.")
    args = parser.parse_args()

    run_dir = args.run_dir
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)

    pairs = collect_pairs(run_dir, gt_index=args.gt_index, prediction_index=args.prediction_index)
    if not pairs:
        raise FileNotFoundError(
            f"No test*-*.png pairs found in {run_dir}. Expected names like test0-0.png and test0-1.png."
        )

    lpips_metric = make_lpips_metric(args.device) if args.lpips else None
    metric_names = ["mse", "rmse", "mae", "pcc", "ssim", "psnr", "cosine"]
    if args.lpips:
        metric_names.append("lpips")

    result_rows = []
    for pair in pairs:
        gt = load_rgb(pair["ground_truth"])
        pred = load_rgb(pair["prediction"])
        if gt.shape != pred.shape:
            raise ValueError(f"Shape mismatch: {pair['ground_truth']} {gt.shape}, {pair['prediction']} {pred.shape}")

        metrics = compute_basic_metrics(gt, pred)
        if lpips_metric is not None:
            metrics["lpips"] = lpips_metric(gt, pred)

        result_rows.append(
            {
                "sample_id": pair["sample_id"],
                "ground_truth": pair["ground_truth"].name,
                "prediction_id": pair["prediction_id"],
                "prediction": pair["prediction"].name,
                **metrics,
            }
        )

    pair_mean, sample_rows, sample_mean = summarize(result_rows, metric_names)

    pair_csv = run_dir / f"{args.output_prefix}_pairs.csv"
    sample_csv = run_dir / f"{args.output_prefix}_samples.csv"
    json_path = run_dir / f"{args.output_prefix}_summary.json"

    write_csv(pair_csv, result_rows, ["sample_id", "ground_truth", "prediction_id", "prediction", *metric_names])
    write_csv(sample_csv, sample_rows, ["sample_id", "num_predictions", *metric_names])

    payload = {
        "run_dir": str(run_dir),
        "num_pairs": len(result_rows),
        "num_samples": len(sample_rows),
        "metrics": metric_names,
        "pair_mean": pair_mean,
        "sample_mean": sample_mean,
        "pairs_csv": str(pair_csv),
        "samples_csv": str(sample_csv),
    }
    json_path.write_text(json.dumps(payload, indent=2, default=json_ready))

    print(f"Run dir: {run_dir}")
    print(f"Samples: {len(sample_rows)}")
    print(f"Pairs: {len(result_rows)}")
    print("\nMean over all generated image pairs:")
    for name in metric_names:
        print(f"  {name}: {pair_mean[name]:.6f}")
    print("\nMean over samples, after averaging generated copies within each sample:")
    for name in metric_names:
        print(f"  {name}: {sample_mean[name]:.6f}")
    print(f"\nSaved: {pair_csv}")
    print(f"Saved: {sample_csv}")
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
