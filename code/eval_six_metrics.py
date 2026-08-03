import argparse
import csv
import json
import math
import platform
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - tqdm is optional for argument inspection.
    def tqdm(x, **kwargs):
        return x


TEST_IMAGE_RE = re.compile(r"^test(?P<sample>\d+)-(?P<copy>\d+)\.(png|jpg|jpeg)$", re.IGNORECASE)
PRED_DIR_RE = re.compile(r"^(prediction|pred|copy|sample)[_-]?(?P<pred>\d+)$", re.IGNORECASE)
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


class EvaluationError(RuntimeError):
    pass


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compute FID, Inception Score, PSNR, SSIM, LPIPS, and 50-way Top-1 "
            "for DreamDiffusion generated images."
        )
    )
    parser.add_argument("--generated_root", type=Path, required=True,
                        help="Generated image root. Supports testX-Y.png files or prediction_1/ subdirectories.")
    parser.add_argument("--ground_truth_dir", type=Path, required=True,
                        help="Ground-truth image directory. For DreamDiffusion outputs this can be the same run dir.")
    parser.add_argument("--output_dir", type=Path, required=True,
                        help="Directory for CSV files and evaluation_config.json.")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--n_way", type=int, default=50)
    parser.add_argument("--num_trials", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0,
                        help="Reserved for compatibility; image loading is deterministic and direct.")
    parser.add_argument("--fid_feature", type=int, default=2048,
                        help="Feature layer for torchmetrics FID. Use 2048 normally, or 64 for comparison.")
    parser.add_argument("--is_splits", type=int, default=10)
    parser.add_argument("--prediction_indices", type=int, nargs="*", default=None,
                        help="Prediction copy ids to evaluate, e.g. --prediction_indices 1 2 3 4 5.")
    parser.add_argument("--gt_index", type=int, default=0,
                        help="DreamDiffusion ground-truth copy id in testX-Y.png files. Default: 0.")
    parser.add_argument("--resize_size", type=int, default=512,
                        help="Resize generated/GT pairs to this square size before PSNR/SSIM/LPIPS/FID/IS.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Evaluate only the first N sample ids per prediction. Useful for quick tests.")
    parser.add_argument("--skip_bad_images", action="store_true",
                        help="Skip unreadable pairs and record them in evaluation_config.json.")
    parser.add_argument("--sanity_check", action="store_true",
                        help="Also run GT-vs-GT and shuffled-GT-vs-GT checks.")
    parser.add_argument("--skip_fid", action="store_true")
    parser.add_argument("--skip_is", action="store_true")
    parser.add_argument("--skip_lpips", action="store_true")
    parser.add_argument("--skip_class", action="store_true")
    args = parser.parse_args()

    if args.batch_size <= 0:
        parser.error("--batch_size must be positive.")
    if args.n_way < 2 or args.n_way > 1000:
        parser.error("--n_way must be in [2, 1000].")
    if args.num_trials <= 0:
        parser.error("--num_trials must be positive.")
    if args.fid_feature <= 0:
        parser.error("--fid_feature must be positive.")
    if args.is_splits <= 0:
        parser.error("--is_splits must be positive.")
    if args.resize_size <= 0:
        parser.error("--resize_size must be positive.")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive.")
    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    except Exception:
        pass


def resolve_device(device_name):
    if device_name == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but unavailable; using CPU.", flush=True)
        return torch.device("cpu")
    return torch.device(device_name)


def list_image_files(path):
    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {path}")
    return sorted([p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])


def sample_id_from_path(path, expected_copy=None):
    match = TEST_IMAGE_RE.match(path.name)
    if match is not None:
        copy_id = int(match.group("copy"))
        if expected_copy is not None and copy_id != expected_copy:
            return None
        return int(match.group("sample"))

    stem = path.stem
    if stem.isdigit():
        return int(stem)
    numbers = re.findall(r"\d+", stem)
    if len(numbers) == 1:
        return int(numbers[0])
    return None


def collect_ground_truth(ground_truth_dir, gt_index):
    result = {}
    ambiguous = []
    for path in list_image_files(ground_truth_dir):
        sample_id = sample_id_from_path(path, expected_copy=gt_index)
        if sample_id is None:
            if TEST_IMAGE_RE.match(path.name):
                continue
            ambiguous.append(path.name)
            continue
        if sample_id in result:
            raise EvaluationError(
                f"Duplicate ground-truth sample_id={sample_id}: {result[sample_id]} and {path}"
            )
        result[sample_id] = path

    if not result:
        hint = f" Ambiguous image names: {ambiguous[:10]}" if ambiguous else ""
        raise EvaluationError(
            f"No ground-truth images found in {ground_truth_dir}. "
            f"Expected testX-{gt_index}.png or numeric filenames like 0.png.{hint}"
        )
    return dict(sorted(result.items()))


def prediction_id_from_dir(path):
    match = PRED_DIR_RE.match(path.name)
    if match is None:
        return None
    return int(match.group("pred"))


def collect_prediction_images(generated_root, gt_index):
    direct_files = list_image_files(generated_root)
    direct_grouped = {}
    for path in direct_files:
        match = TEST_IMAGE_RE.match(path.name)
        if match is None:
            continue
        sample_id = int(match.group("sample"))
        copy_id = int(match.group("copy"))
        if copy_id == gt_index:
            continue
        direct_grouped.setdefault(copy_id, {})[sample_id] = path

    if direct_grouped:
        return {k: dict(sorted(v.items())) for k, v in sorted(direct_grouped.items())}

    prediction_groups = {}
    for child in sorted([p for p in generated_root.iterdir() if p.is_dir()]):
        pred_id = prediction_id_from_dir(child)
        if pred_id is None:
            continue
        images = {}
        for path in list_image_files(child):
            sample_id = sample_id_from_path(path, expected_copy=None)
            if sample_id is None:
                continue
            if sample_id in images:
                raise EvaluationError(
                    f"Duplicate generated sample_id={sample_id} in {child}: {images[sample_id]} and {path}"
                )
            images[sample_id] = path
        if images:
            prediction_groups[pred_id] = dict(sorted(images.items()))

    if not prediction_groups:
        raise EvaluationError(
            f"No generated images found in {generated_root}. Expected DreamDiffusion testX-Y.png files "
            "or subdirectories named prediction_1, prediction_2, ..."
        )
    return dict(sorted(prediction_groups.items()))


def select_prediction_indices(predictions, requested):
    available = sorted(predictions)
    if requested:
        missing = [idx for idx in requested if idx not in predictions]
        if missing:
            raise EvaluationError(
                f"Requested prediction_indices not found: {missing}. Available: {available}"
            )
        return requested
    return available


def load_rgb_float(path, resize_size, skip_bad_images=False):
    try:
        with Image.open(path) as img:
            img = img.convert("RGB")
            if resize_size is not None:
                img = img.resize((resize_size, resize_size), Image.BILINEAR)
            arr = np.asarray(img, dtype=np.float32) / 255.0
            return arr
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        if skip_bad_images:
            return None
        raise EvaluationError(f"Failed to read image {path}: {type(exc).__name__}: {exc}") from exc


def align_pairs(gt_map, pred_map, limit, resize_size, skip_bad_images):
    pred_ids = sorted(pred_map)
    if limit is not None:
        pred_ids = pred_ids[:limit]

    missing_gt = [sample_id for sample_id in pred_ids if sample_id not in gt_map]
    if missing_gt:
        raise EvaluationError(
            f"Generated images have no matching ground truth for sample ids: {missing_gt[:30]}. "
            f"First missing expected from ground_truth_dir: {[f'test{x}-0.png or {x}.png' for x in missing_gt[:10]]}"
        )

    pairs = []
    skipped = []
    for sample_id in pred_ids:
        pred_path = pred_map[sample_id]
        gt_path = gt_map[sample_id]
        pred = load_rgb_float(pred_path, resize_size, skip_bad_images)
        gt = load_rgb_float(gt_path, resize_size, skip_bad_images)
        if pred is None or gt is None:
            skipped.append({"sample_id": sample_id, "generated_path": str(pred_path), "ground_truth_path": str(gt_path)})
            continue
        pairs.append({
            "sample_id": sample_id,
            "generated_path": pred_path,
            "ground_truth_path": gt_path,
            "generated": pred,
            "ground_truth": gt,
        })

    if not pairs:
        raise EvaluationError("No valid image pairs after loading. Check images or --skip_bad_images.")
    return pairs, skipped


def np_to_nchw_float(images):
    arr = np.stack(images, axis=0).astype(np.float32)
    return torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()


def batch_iter(items, batch_size):
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def finite_mean_std(values):
    arr = np.asarray(values, dtype=np.float64)
    if np.isinf(arr).any():
        return float("inf"), 0.0
    return float(np.mean(arr)), float(np.std(arr, ddof=0))


def compute_psnr(pred, gt):
    mse = np.mean((pred - gt) ** 2)
    if mse == 0:
        return float("inf")
    return float(10.0 * np.log10(1.0 / mse))


def compute_ssim(pred, gt):
    try:
        from skimage.metrics import structural_similarity
    except Exception as exc:
        raise EvaluationError(
            "SSIM requires scikit-image. Install with: pip install scikit-image"
        ) from exc
    return float(structural_similarity(gt, pred, channel_axis=-1, data_range=1.0))


@torch.no_grad()
def compute_lpips_scores(pairs, device, batch_size):
    try:
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    except Exception as exc:
        raise EvaluationError(
            "LPIPS requires torchmetrics with image dependencies. Try: pip install torchmetrics lpips"
        ) from exc

    try:
        metric = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=False, reduction="none").to(device)
    except TypeError:
        try:
            metric = LearnedPerceptualImagePatchSimilarity(net_type="alex", reduction="none").to(device)
        except TypeError:
            metric = LearnedPerceptualImagePatchSimilarity(net_type="alex").to(device)
    metric.eval()

    scores = []
    for batch in tqdm(list(batch_iter(pairs, batch_size)), desc="LPIPS", leave=False):
        pred = np_to_nchw_float([x["generated"] for x in batch]).to(device) * 2.0 - 1.0
        gt = np_to_nchw_float([x["ground_truth"] for x in batch]).to(device) * 2.0 - 1.0
        value = metric(pred, gt)
        if value.ndim == 0:
            if len(batch) == 1:
                scores.append(float(value.detach().cpu()))
            else:
                # Some torchmetrics versions reduce by mean. Fall back to single-image calls for per-sample CSV.
                for pred_one, gt_one in zip(pred, gt):
                    scores.append(float(metric(pred_one.unsqueeze(0), gt_one.unsqueeze(0)).detach().cpu()))
        else:
            scores.extend([float(v) for v in value.detach().cpu().flatten()])

    del metric
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return scores


@torch.no_grad()
def compute_fid(pairs, device, batch_size, feature):
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
    except Exception as exc:
        raise EvaluationError(
            "FID requires torchmetrics image dependencies and torch-fidelity. "
            "Try: pip install torchmetrics torch-fidelity"
        ) from exc

    metric = FrechetInceptionDistance(feature=feature, normalize=True).to(device)
    metric.eval()
    for batch in tqdm(list(batch_iter(pairs, batch_size)), desc="FID real", leave=False):
        gt = np_to_nchw_float([x["ground_truth"] for x in batch]).to(device)
        metric.update(gt, real=True)
    for batch in tqdm(list(batch_iter(pairs, batch_size)), desc="FID generated", leave=False):
        pred = np_to_nchw_float([x["generated"] for x in batch]).to(device)
        metric.update(pred, real=False)
    value = float(metric.compute().detach().cpu())
    del metric
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return value


@torch.no_grad()
def compute_inception_score(pairs, device, batch_size, requested_splits):
    try:
        from torchmetrics.image.inception import InceptionScore
    except Exception as exc:
        raise EvaluationError(
            "Inception Score requires torchmetrics image dependencies and torch-fidelity. "
            "Try: pip install torchmetrics torch-fidelity"
        ) from exc

    splits = max(1, min(requested_splits, len(pairs)))
    if splits != requested_splits:
        print(f"[WARN] IS splits reduced from {requested_splits} to {splits} for {len(pairs)} images.", flush=True)
    metric = InceptionScore(splits=splits, normalize=True).to(device)
    metric.eval()
    for batch in tqdm(list(batch_iter(pairs, batch_size)), desc="Inception Score", leave=False):
        pred = np_to_nchw_float([x["generated"] for x in batch]).to(device)
        metric.update(pred)
    mean, std = metric.compute()
    mean = float(mean.detach().cpu())
    std = float(std.detach().cpu())
    if math.isnan(std):
        std = 0.0
    del metric
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return mean, std


@torch.no_grad()
def compute_vit_logits(paths, device, batch_size):
    try:
        from torchvision.models import ViT_H_14_Weights, vit_h_14
    except Exception as exc:
        raise EvaluationError("50-way Top-1 requires torchvision with vit_h_14 support.") from exc

    weights = ViT_H_14_Weights.DEFAULT
    maybe_print_weight_cache("ViT-H-14", getattr(weights, "url", None))
    model = vit_h_14(weights=weights).to(device)
    model.eval()
    preprocess = weights.transforms()

    logits = []
    for batch_paths in tqdm(list(batch_iter(paths, batch_size)), desc="ViT-H-14 logits", leave=False):
        images = []
        for path in batch_paths:
            with Image.open(path) as img:
                images.append(preprocess(img.convert("RGB")))
        batch = torch.stack(images, dim=0).to(device)
        logits.append(model(batch).detach().cpu())

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return torch.cat(logits, dim=0)


def maybe_print_weight_cache(label, url):
    if not url:
        return
    filename = Path(url).name
    cache_path = Path(torch.hub.get_dir()) / "checkpoints" / filename
    if cache_path.exists():
        print(f"{label} cache found: {cache_path}", flush=True)
    else:
        print(f"[WARN] {label} cache not found at: {cache_path}", flush=True)
        print(f"[WARN] torchvision/torchmetrics may try to download: {url}", flush=True)


def compute_50way(pairs, device, batch_size, n_way, num_trials, seed):
    print(f"50-way random baseline: 1 / {n_way} = {1.0 / n_way:.4f} ({100.0 / n_way:.2f}%)", flush=True)
    gt_paths = [x["ground_truth_path"] for x in pairs]
    pred_paths = [x["generated_path"] for x in pairs]
    gt_logits = compute_vit_logits(gt_paths, device, batch_size)
    pred_logits = compute_vit_logits(pred_paths, device, batch_size)
    gt_classes = gt_logits.argmax(dim=1).numpy().astype(int)
    pred_top1 = pred_logits.argmax(dim=1).numpy().astype(int)
    pred_logits_np = pred_logits.numpy()
    rng = np.random.default_rng(seed)

    sample_scores = []
    trial_stds = []
    for row_idx, gt_class in enumerate(gt_classes):
        wrong_classes = np.array([x for x in range(pred_logits_np.shape[1]) if x != gt_class], dtype=np.int64)
        trials = []
        for _ in range(num_trials):
            negatives = rng.choice(wrong_classes, size=n_way - 1, replace=False)
            candidates = np.concatenate([[gt_class], negatives])
            winner = int(np.argmax(pred_logits_np[row_idx, candidates]))
            trials.append(1.0 if winner == 0 else 0.0)
        sample_scores.append(float(np.mean(trials)))
        trial_stds.append(float(np.std(trials, ddof=0)))

    return sample_scores, trial_stds, gt_classes.tolist(), pred_top1.tolist()


def check_no_nan(row, allow_missing=False):
    for key, value in row.items():
        if value == "" or value is None:
            if allow_missing:
                continue
            raise EvaluationError(f"Missing value for {key}.")
        if isinstance(value, float) and math.isnan(value):
            if key.endswith("_std") and math.isinf(row.get(key.replace("_std", "_mean"), 0.0)):
                continue
            if allow_missing:
                continue
            raise EvaluationError(f"NaN value for {key}.")


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def append_and_write(path, rows, fieldnames):
    write_csv(path, rows, fieldnames)


def evaluate_one_prediction(prediction_index, pairs, args, device):
    print("", flush=True)
    print(f"prediction_index={prediction_index}", flush=True)
    print(f"num paired samples: {len(pairs)}", flush=True)
    if len(pairs) < 10:
        print("[WARN] FID/IS are high-variance with very few images.", flush=True)

    per_sample = []
    for item in tqdm(pairs, desc="PSNR/SSIM", leave=False):
        psnr = compute_psnr(item["generated"], item["ground_truth"])
        ssim = compute_ssim(item["generated"], item["ground_truth"])
        per_sample.append({
            "prediction_index": prediction_index,
            "sample_id": item["sample_id"],
            "generated_path": str(item["generated_path"]),
            "ground_truth_path": str(item["ground_truth_path"]),
            "psnr": psnr,
            "ssim": ssim,
            "lpips": "",
            "top1_50way": "",
            "top1_50way_percent": "",
            "top1_50way_trial_std": "",
            "gt_pseudo_class": "",
            "generated_top1_imagenet_class": "",
        })

    psnr_values = [row["psnr"] for row in per_sample]
    ssim_values = [row["ssim"] for row in per_sample]
    psnr_mean, psnr_std = finite_mean_std(psnr_values)
    ssim_mean, ssim_std = finite_mean_std(ssim_values)

    lpips_mean = lpips_std = ""
    if not args.skip_lpips:
        print("start metric: LPIPS alex", flush=True)
        lpips_values = compute_lpips_scores(pairs, device, args.batch_size)
        for row, value in zip(per_sample, lpips_values):
            row["lpips"] = value
        lpips_mean, lpips_std = finite_mean_std(lpips_values)

    top1_mean = top1_percent = top1_std = ""
    if not args.skip_class:
        print("start metric: 50-way Top-1 Accuracy", flush=True)
        top1_values, trial_stds, gt_classes, pred_classes = compute_50way(
            pairs, device, args.batch_size, args.n_way, args.num_trials, args.seed + int(prediction_index)
            if isinstance(prediction_index, int) else args.seed
        )
        for row, acc, trial_std, gt_class, pred_class in zip(per_sample, top1_values, trial_stds, gt_classes, pred_classes):
            row["top1_50way"] = acc
            row["top1_50way_percent"] = acc * 100.0
            row["top1_50way_trial_std"] = trial_std
            row["gt_pseudo_class"] = gt_class
            row["generated_top1_imagenet_class"] = pred_class
        top1_mean, top1_std = finite_mean_std(top1_values)
        top1_percent = top1_mean * 100.0

    fid = ""
    if not args.skip_fid:
        print(f"start metric: FID feature={args.fid_feature}", flush=True)
        fid = compute_fid(pairs, device, args.batch_size, args.fid_feature)

    is_mean = is_std = ""
    if not args.skip_is:
        print("start metric: Inception Score", flush=True)
        is_mean, is_std = compute_inception_score(pairs, device, args.batch_size, args.is_splits)

    row = {
        "prediction_index": prediction_index,
        "num_samples": len(pairs),
        "fid": fid,
        "fid_feature": args.fid_feature,
        "inception_score_mean": is_mean,
        "inception_score_std": is_std,
        "psnr_mean": psnr_mean,
        "psnr_std": psnr_std,
        "ssim_mean": ssim_mean,
        "ssim_std": ssim_std,
        "lpips_mean": lpips_mean,
        "lpips_std": lpips_std,
        "top1_50way": top1_mean,
        "top1_50way_percent": top1_percent,
        "top1_50way_std": top1_std,
        "n_way": args.n_way,
        "num_trials": args.num_trials,
        "seed": args.seed,
    }
    check_no_nan(row, allow_missing=True)
    for sample_row in per_sample:
        check_no_nan(sample_row, allow_missing=True)
    print(f"metrics result: {row}", flush=True)
    return row, per_sample


def summary_rows(prediction_rows):
    specs = [
        ("FID", "fid", "lower"),
        ("IS", "inception_score_mean", "higher"),
        ("PSNR", "psnr_mean", "higher"),
        ("SSIM", "ssim_mean", "higher"),
        ("LPIPS", "lpips_mean", "lower"),
        ("50-way Top-1 Accuracy", "top1_50way", "higher"),
    ]
    rows = []
    for name, key, direction in specs:
        vals = []
        for row in prediction_rows:
            value = row.get(key, "")
            if value == "" or value is None:
                continue
            vals.append(float(value))
        if vals:
            rows.append({
                "metric": name,
                "mean_across_predictions": float(np.mean(vals)),
                "std_across_predictions": float(np.std(vals, ddof=0)),
                "direction": direction,
            })
        else:
            rows.append({
                "metric": name,
                "mean_across_predictions": "",
                "std_across_predictions": "",
                "direction": direction,
            })
    return rows


def package_versions():
    versions = {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "torch": getattr(torch, "__version__", None),
        "cuda": getattr(torch.version, "cuda", None),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    try:
        import torchvision
        versions["torchvision"] = torchvision.__version__
    except Exception as exc:
        versions["torchvision"] = f"unavailable: {type(exc).__name__}: {exc}"
    try:
        import torchmetrics
        versions["torchmetrics"] = torchmetrics.__version__
    except Exception as exc:
        versions["torchmetrics"] = f"unavailable: {type(exc).__name__}: {exc}"
    try:
        import lpips
        versions["lpips"] = getattr(lpips, "__version__", "installed")
    except Exception as exc:
        versions["lpips"] = f"unavailable: {type(exc).__name__}: {exc}"
    return versions


def run_sanity_checks(gt_map, args, device):
    gt_source = sorted(gt_map.items())
    if args.limit is not None:
        gt_source = gt_source[:args.limit]
    gt_items = [{"sample_id": sid, "generated_path": path, "ground_truth_path": path,
                 "generated": load_rgb_float(path, args.resize_size, args.skip_bad_images),
                 "ground_truth": load_rgb_float(path, args.resize_size, args.skip_bad_images)}
                for sid, path in gt_source]
    gt_items = [x for x in gt_items if x["generated"] is not None and x["ground_truth"] is not None]
    if len(gt_items) < 2:
        print("[WARN] sanity_check needs at least 2 GT images for shuffled check.", flush=True)
    gt_row, _ = evaluate_one_prediction("sanity_gt_vs_gt", gt_items, args, device)

    shuffled = list(gt_items)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(shuffled))
    shuffled_items = []
    for target_item, source_idx in zip(gt_items, perm):
        source_item = gt_items[int(source_idx)]
        shuffled_items.append({
            "sample_id": target_item["sample_id"],
            "generated_path": source_item["ground_truth_path"],
            "ground_truth_path": target_item["ground_truth_path"],
            "generated": source_item["ground_truth"],
            "ground_truth": target_item["ground_truth"],
        })
    shuf_row, _ = evaluate_one_prediction("sanity_shuffled_gt_vs_gt", shuffled_items, args, device)

    warnings = []
    if gt_row.get("ssim_mean", 0) != "" and float(gt_row["ssim_mean"]) < 0.99:
        warnings.append("GT-vs-GT SSIM is below 0.99; check preprocessing or pairing.")
    if gt_row.get("lpips_mean", "") != "" and float(gt_row["lpips_mean"]) > 0.01:
        warnings.append("GT-vs-GT LPIPS is above 0.01; check LPIPS input range.")
    if gt_row.get("top1_50way", "") != "" and float(gt_row["top1_50way"]) < 0.99:
        warnings.append("GT-vs-GT 50-way Top-1 is below 0.99; check class logic.")
    for warning in warnings:
        print(f"[SANITY WARNING] {warning}", flush=True)
    return [gt_row, shuf_row]


def main():
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"generated_root: {args.generated_root.resolve()}", flush=True)
    print(f"ground_truth_dir: {args.ground_truth_dir.resolve()}", flush=True)
    print(f"output_dir: {args.output_dir.resolve()}", flush=True)
    print(f"device: {device}", flush=True)
    print(f"50-way random baseline: 1 / {args.n_way} = {1.0 / args.n_way:.4f}", flush=True)

    gt_map = collect_ground_truth(args.ground_truth_dir, args.gt_index)
    predictions = collect_prediction_images(args.generated_root, args.gt_index)
    prediction_indices = select_prediction_indices(predictions, args.prediction_indices)
    print(f"ground-truth images found: {len(gt_map)}", flush=True)
    print(f"prediction indices found: {sorted(predictions)}", flush=True)
    print(f"prediction indices selected: {prediction_indices}", flush=True)

    by_prediction_path = args.output_dir / "metrics_by_prediction.csv"
    per_sample_path = args.output_dir / "metrics_per_sample.csv"
    summary_path = args.output_dir / "metrics_summary.csv"
    sanity_path = args.output_dir / "sanity_check_metrics.csv"

    prediction_fields = [
        "prediction_index", "num_samples", "fid", "fid_feature", "inception_score_mean",
        "inception_score_std", "psnr_mean", "psnr_std", "ssim_mean", "ssim_std",
        "lpips_mean", "lpips_std", "top1_50way", "top1_50way_percent",
        "top1_50way_std", "n_way", "num_trials", "seed",
    ]
    sample_fields = [
        "prediction_index", "sample_id", "generated_path", "ground_truth_path", "psnr",
        "ssim", "lpips", "top1_50way", "top1_50way_percent", "top1_50way_trial_std",
        "gt_pseudo_class", "generated_top1_imagenet_class",
    ]
    summary_fields = ["metric", "mean_across_predictions", "std_across_predictions", "direction"]

    prediction_rows = []
    per_sample_rows = []
    generated_counts = {}
    skipped_bad_images = []
    started_at = datetime.now().isoformat(timespec="seconds")
    start_time = time.time()

    for prediction_index in prediction_indices:
        pairs, skipped = align_pairs(
            gt_map, predictions[prediction_index], args.limit, args.resize_size, args.skip_bad_images
        )
        skipped_bad_images.extend(skipped)
        generated_counts[str(prediction_index)] = len(predictions[prediction_index])
        row, sample_rows = evaluate_one_prediction(prediction_index, pairs, args, device)
        prediction_rows.append(row)
        per_sample_rows.extend(sample_rows)
        append_and_write(by_prediction_path, prediction_rows, prediction_fields)
        append_and_write(per_sample_path, per_sample_rows, sample_fields)
        append_and_write(summary_path, summary_rows(prediction_rows), summary_fields)

    sanity_rows = []
    if args.sanity_check:
        sanity_rows = run_sanity_checks(gt_map, args, device)
        write_csv(sanity_path, sanity_rows, prediction_fields)

    config = {
        "argv": sys.argv,
        "generated_root": str(args.generated_root.resolve()),
        "ground_truth_dir": str(args.ground_truth_dir.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "evaluation_time": started_at,
        "elapsed_seconds": round(time.time() - start_time, 3),
        "package_versions": package_versions(),
        "ground_truth_image_count": len(gt_map),
        "generated_image_counts": generated_counts,
        "prediction_indices": prediction_indices,
        "classifier": "torchvision.models.vit_h_14",
        "classifier_weights": "ViT_H_14_Weights.DEFAULT",
        "fid_feature": args.fid_feature,
        "resize_and_normalization": {
            "resize_size": args.resize_size,
            "rgb": True,
            "psnr_ssim_fid_is_range": "float [0, 1]",
            "lpips_range": "float [-1, 1]",
            "vit_preprocess": "ViT_H_14_Weights.DEFAULT.transforms()",
        },
        "n_way": args.n_way,
        "num_trials": args.num_trials,
        "seed": args.seed,
        "skip_fid": args.skip_fid,
        "skip_is": args.skip_is,
        "skip_lpips": args.skip_lpips,
        "skip_class": args.skip_class,
        "skip_bad_images": args.skip_bad_images,
        "skipped_bad_images": skipped_bad_images,
    }
    (args.output_dir / "evaluation_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False))

    print("", flush=True)
    print(f"saved: {by_prediction_path}", flush=True)
    print(f"saved: {per_sample_path}", flush=True)
    print(f"saved: {summary_path}", flush=True)
    print(f"saved: {args.output_dir / 'evaluation_config.json'}", flush=True)
    if args.sanity_check:
        print(f"saved: {sanity_path}", flush=True)


if __name__ == "__main__":
    main()
