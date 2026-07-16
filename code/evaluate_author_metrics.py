import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image


CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))


TEST_IMAGE_RE = re.compile(r"^test(?P<sample>\d+)-(?P<copy>\d+)\.png$")


def patch_torchmetrics_accuracy(author_eval_metrics):
    """Keep the author's old torchmetrics accuracy call working on newer torchmetrics."""
    original_accuracy = author_eval_metrics.accuracy

    def accuracy_compat(preds, target, top_k=1):
        try:
            return original_accuracy(preds, target, top_k=top_k)
        except TypeError:
            return original_accuracy(
                preds,
                target,
                task="multiclass",
                num_classes=preds.shape[-1],
                top_k=top_k,
            )

    author_eval_metrics.accuracy = accuracy_compat


def load_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)


def collect_generated_samples(run_dir, gt_index):
    grouped = {}
    for path in run_dir.glob("test*-*.png"):
        match = TEST_IMAGE_RE.match(path.name)
        if match is None:
            continue
        sample_id = int(match.group("sample"))
        copy_id = int(match.group("copy"))
        grouped.setdefault(sample_id, {})[copy_id] = path

    samples = {}
    for sample_id, copies in grouped.items():
        if gt_index in copies:
            samples[sample_id] = copies
    return dict(sorted(samples.items()))


def select_prediction_indices(samples, gt_index, prediction_index):
    if prediction_index is not None:
        return [prediction_index]

    pred_ids = set()
    for copies in samples.values():
        pred_ids.update(copy_id for copy_id in copies if copy_id != gt_index)
    return sorted(pred_ids)


def stack_for_prediction(samples, gt_index, pred_index):
    sample_ids = []
    gt_images = []
    pred_images = []

    for sample_id, copies in samples.items():
        if gt_index not in copies or pred_index not in copies:
            continue
        sample_ids.append(sample_id)
        gt_images.append(load_rgb(copies[gt_index]))
        pred_images.append(load_rgb(copies[pred_index]))

    if not gt_images:
        raise FileNotFoundError(f"No complete image pairs found for prediction index {pred_index}.")

    return sample_ids, np.stack(gt_images), np.stack(pred_images)


def limit_samples(samples, limit):
    if limit is None:
        return samples
    return dict(list(samples.items())[:limit])


def finite_mean(values):
    clean = [value for value in values if not (isinstance(value, float) and math.isnan(value))]
    if not clean:
        return float("nan")
    return float(np.mean(clean))


def write_csv(path, rows, fieldnames):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def json_ready(value):
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate DreamDiffusion outputs with the author's eval_metrics.py logic."
    )
    parser.add_argument("run_dir", type=Path, help="A results/eval/<timestamp> directory.")
    parser.add_argument("--gt-index", type=int, default=0, help="Ground-truth image index. Default: 0.")
    parser.add_argument(
        "--prediction-index",
        type=int,
        default=None,
        help="Evaluate only one generated copy, e.g. 1. Default: average all generated copies.",
    )
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Device for PSM and class metrics.")
    parser.add_argument("--n-way", type=int, default=50, help="Author-style n-way classification setting.")
    parser.add_argument("--num-trials", type=int, default=50, help="Number of random trials for n-way top-k accuracy.")
    parser.add_argument("--top-k", type=int, default=1, help="Top-k for n-way classification accuracy.")
    parser.add_argument("--skip-psm", action="store_true", help="Skip LPIPS/PSM pair-wise metric.")
    parser.add_argument("--skip-class", action="store_true", help="Skip author-style 50-way top-1 class metric.")
    parser.add_argument(
        "--metrics",
        default="mse,pcc,ssim",
        help="Comma-separated pair-wise metrics to run. Choices: mse,pcc,ssim,psm. Default: mse,pcc,ssim.",
    )
    parser.add_argument(
        "--limit-samples",
        type=int,
        default=None,
        help="Evaluate only the first N samples for a quick smoke test.",
    )
    parser.add_argument("--output-prefix", default="author_eval", help="Output file prefix.")
    args = parser.parse_args()

    if not args.run_dir.exists():
        raise FileNotFoundError(args.run_dir)

    print("Importing author eval_metrics.py ...", flush=True)
    import eval_metrics as author_eval_metrics

    patch_torchmetrics_accuracy(author_eval_metrics)

    samples = collect_generated_samples(args.run_dir, args.gt_index)
    samples = limit_samples(samples, args.limit_samples)
    if not samples:
        raise FileNotFoundError(
            f"No test images found in {args.run_dir}. Expected files like test0-0.png and test0-1.png."
        )

    prediction_indices = select_prediction_indices(samples, args.gt_index, args.prediction_index)
    if not prediction_indices:
        raise FileNotFoundError("No generated prediction images found.")

    pairwise_metrics = [metric.strip() for metric in args.metrics.split(",") if metric.strip()]
    unknown_metrics = sorted(set(pairwise_metrics) - {"mse", "pcc", "ssim", "psm"})
    if unknown_metrics:
        raise ValueError(f"Unknown metrics: {unknown_metrics}. Valid choices are mse,pcc,ssim,psm.")
    if args.skip_psm:
        pairwise_metrics = [metric for metric in pairwise_metrics if metric != "psm"]

    print(f"Run dir: {args.run_dir}", flush=True)
    print(f"Samples with ground truth: {len(samples)}", flush=True)
    print(f"Prediction indices: {prediction_indices}", flush=True)
    print(f"Pair-wise metrics: {pairwise_metrics}", flush=True)
    print(f"Class metric enabled: {not args.skip_class}", flush=True)

    rows = []
    for pred_index in prediction_indices:
        print(f"\nLoading images for prediction index {pred_index} ...", flush=True)
        sample_ids, gt_images, pred_images = stack_for_prediction(samples, args.gt_index, pred_index)
        row = {
            "prediction_index": pred_index,
            "num_samples": len(sample_ids),
        }

        for metric_name in pairwise_metrics:
            print(
                f"Computing author pair-wise_{metric_name} for prediction index {pred_index} "
                f"({len(sample_ids)} samples; this compares each prediction against all other GT images) ...",
                flush=True,
            )
            scores = author_eval_metrics.get_similarity_metric(
                pred_images,
                gt_images,
                method="pair-wise",
                metric_name=metric_name,
            )
            row[f"pair-wise_{metric_name}"] = finite_mean(scores)
            print(f"Finished pair-wise_{metric_name}: {row[f'pair-wise_{metric_name}']:.6f}", flush=True)

        if not args.skip_class:
            print(
                f"Computing author {args.n_way}-way top-{args.top_k} class metric for prediction index {pred_index} ...",
                flush=True,
            )
            scores = author_eval_metrics.get_similarity_metric(
                pred_images,
                gt_images,
                method="class",
                metric_name=None,
                n_way=args.n_way,
                num_trials=args.num_trials,
                top_k=args.top_k,
                device=args.device,
            )
            row[f"top-{args.top_k}-class"] = finite_mean(scores)
            print(f"Finished top-{args.top_k}-class: {row[f'top-{args.top_k}-class']:.6f}", flush=True)

        rows.append(row)

    summary = {
        "run_dir": str(args.run_dir),
        "num_prediction_indices": len(prediction_indices),
        "prediction_indices": prediction_indices,
        "gt_index": args.gt_index,
        "n_way": args.n_way,
        "num_trials": args.num_trials,
        "top_k": args.top_k,
        "pairwise_metrics": pairwise_metrics,
        "skip_class": args.skip_class,
        "per_prediction": rows,
    }

    for metric_name in pairwise_metrics:
        key = f"pair-wise_{metric_name}"
        summary[key] = finite_mean([row[key] for row in rows])

    class_key = f"top-{args.top_k}-class"
    if not args.skip_class:
        class_scores = [row[class_key] for row in rows]
        summary[class_key] = finite_mean(class_scores)
        summary[f"{class_key} (max)"] = float(np.max(class_scores))

    csv_path = args.run_dir / f"{args.output_prefix}_per_prediction.csv"
    json_path = args.run_dir / f"{args.output_prefix}_summary.json"
    fieldnames = list(rows[0].keys())
    write_csv(csv_path, rows, fieldnames)
    json_path.write_text(json.dumps(summary, indent=2, default=json_ready))

    print(f"Run dir: {args.run_dir}")
    print(f"Prediction indices: {prediction_indices}")
    print(f"Samples with ground truth: {len(samples)}")
    print("\nAuthor-style summary:")
    for metric_name in pairwise_metrics:
        key = f"pair-wise_{metric_name}"
        print(f"  {key}: {summary[key]:.6f}")
    if not args.skip_class:
        print(f"  {class_key}: {summary[class_key]:.6f}")
        print(f"  {class_key} (max): {summary[f'{class_key} (max)']:.6f}")
    print(f"\nSaved: {csv_path}")
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
