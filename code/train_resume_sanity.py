import argparse
import copy
import os
from pathlib import Path


def normalize(img):
    import torch
    from einops import rearrange

    if img.shape[-1] == 3:
        img = rearrange(img, "h w c -> c h w")
    img = torch.tensor(img)
    img = img * 2.0 - 1.0
    return img


def channel_last(img):
    from einops import rearrange

    if img.shape[-1] == 3:
        return img
    return rearrange(img, "c h w -> h w c")


class random_crop:
    def __init__(self, size, p):
        self.size = size
        self.p = p

    def __call__(self, img):
        import torch
        import torchvision.transforms as transforms

        if torch.rand(1) < self.p:
            return transforms.RandomCrop(size=(self.size, self.size))(img)
        return img


def parse_args():
    parser = argparse.ArgumentParser(
        description="Resume DreamDiffusion generation checkpoint for a tiny sanity training run."
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--dataset", type=str, default="EEG")
    parser.add_argument("--pretrained_generation_checkpoint", "--resume_from_checkpoint", dest="checkpoint", type=Path,
                        required=True, help="Official or previous DreamDiffusion generation checkpoint.")
    parser.add_argument("--eeg_signals_path", type=Path, default=None)
    parser.add_argument("--splits_path", type=Path, default=None)
    parser.add_argument("--imagenet_path", type=Path, default=None)
    parser.add_argument("--pretrain_root", type=Path, default=None,
                        help="Directory containing models/config15.yaml.")
    parser.add_argument("--config_patch", type=Path, default=None,
                        help="Stable Diffusion config path. Defaults to pretrain_root/models/config15.yaml.")
    parser.add_argument("--output_dir", type=Path, default=Path("checkpoints/sanity_resume"))
    parser.add_argument("--max_steps", type=int, default=20)
    parser.add_argument("--save_every_n_steps", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--subject", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--disable_clip_tune", action="store_true",
                        help="Debug escape hatch. By default this keeps the checkpoint config's clip_tune value.")
    parser.add_argument("--disable_cls_tune", action="store_true",
                        help="Debug escape hatch. By default this keeps the checkpoint config's cls_tune value.")
    return parser.parse_args()


def require_file(path, description):
    if path is None:
        raise FileNotFoundError(f"{description} path is not set.")
    if not Path(path).exists():
        raise FileNotFoundError(f"{description} not found: {path}")


def resolve_path(root, path):
    if path is None:
        return None
    path = Path(path)
    if path.is_absolute():
        return path
    return root / path


def move_to_device(value, device):
    import torch
    from collections.abc import Mapping

    if torch.is_tensor(value):
        return value.to(device)
    if hasattr(value, "to") and callable(value.to):
        return value.to(device)
    if isinstance(value, Mapping):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    return value


def tensor_to_float(value):
    import torch

    if torch.is_tensor(value):
        return float(value.detach().cpu())
    return float(value)


def check_batch_devices(batch, device):
    import torch
    from collections.abc import Mapping

    problems = []

    def visit(value, path):
        if torch.is_tensor(value):
            if value.device != device:
                problems.append(f"{path}: {value.device}")
            return
        if isinstance(value, Mapping):
            for key, item in value.items():
                visit(item, f"{path}.{key}")

    visit(batch, "batch")
    if problems:
        raise RuntimeError(
            "Some batch tensors are not on the requested device "
            f"{device}: {problems[:20]}"
        )


def make_train_transform(config):
    import torchvision.transforms as transforms

    crop_pix = int(config.crop_ratio * config.img_size)
    return transforms.Compose([
        normalize,
        transforms.Resize((512, 512)),
        random_crop(config.img_size - crop_pix, p=0.5),
        transforms.Resize((512, 512)),
        channel_last,
    ])


def prepare_config(args, checkpoint_payload):
    from config import Config_Generative_Model

    if "config" in checkpoint_payload:
        config = copy.deepcopy(checkpoint_payload["config"])
    else:
        config = Config_Generative_Model()

    root = args.root.resolve()
    config.root_path = str(root)
    config.output_path = str(args.output_dir)
    config.dataset = args.dataset
    config.logger = None
    config.trainer = None

    config.eeg_signals_path = str(resolve_path(root, args.eeg_signals_path) if args.eeg_signals_path else root / "datasets/eeg_5_95_std.pth")
    config.splits_path = str(resolve_path(root, args.splits_path) if args.splits_path else root / "datasets/block_splits_by_image_single.pth")
    config.pretrain_gm_path = str(resolve_path(root, args.pretrain_root) if args.pretrain_root else root / "pretrains")

    defaults = {
        "clip_tune": True,
        "cls_tune": False,
        "eval_avg": True,
        "global_pool": False,
        "use_time_cond": True,
        "crop_ratio": 0.2,
        "img_size": 512,
        "ddim_steps": 250,
        "HW": None,
    }
    for key, value in defaults.items():
        if not hasattr(config, key):
            setattr(config, key, value)

    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.lr is not None:
        config.lr = args.lr
    if args.subject is not None:
        config.subject = args.subject
    elif not hasattr(config, "subject"):
        config.subject = 4
    if args.seed is not None:
        config.seed = args.seed
    if args.disable_clip_tune:
        config.clip_tune = False
    if args.disable_cls_tune:
        config.cls_tune = False

    return config


def build_optimizer(model):
    import torch

    lr = model.learning_rate
    params = []
    if getattr(model, "train_cond_stage_only", False):
        print(f"{model.__class__.__name__}: Only optimizing conditioner params with duplicate filtering!", flush=True)
        params.extend(model.cond_stage_model.parameters())
        for name, param in model.named_parameters():
            if name.startswith("cond_stage_model."):
                continue
            if "attn2" in name or "time_embed_condtion" in name or "norm2" in name:
                params.append(param)
    else:
        params.extend(model.model.parameters())
        if getattr(model, "cond_stage_trainable", False):
            params.extend(model.cond_stage_model.parameters())
        if getattr(model, "learn_logvar", False):
            params.append(model.logvar)

    unique_params = []
    seen = set()
    duplicate_count = 0
    for param in params:
        if not param.requires_grad:
            param.requires_grad = True
        param_id = id(param)
        if param_id in seen:
            duplicate_count += 1
            continue
        seen.add(param_id)
        unique_params.append(param)

    if not unique_params:
        raise RuntimeError("No trainable parameters were selected for the optimizer.")

    print(
        f"optimizer params: {len(unique_params)} unique tensors"
        + (f", filtered duplicates: {duplicate_count}" if duplicate_count else ""),
        flush=True,
    )
    return torch.optim.AdamW(unique_params, lr=lr)


def save_checkpoint(path, model, config, optimizer, global_step, source_checkpoint, device):
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    config_to_save = copy.deepcopy(config)
    config_to_save.logger = None
    config_to_save.trainer = None

    if device.type == "cuda":
        rng_state = torch.cuda.get_rng_state(device)
    else:
        rng_state = torch.random.get_rng_state()

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config_to_save,
            "state": rng_state,
            "optimizer_state_dict": optimizer.state_dict(),
            "global_step": global_step,
            "source_checkpoint": str(source_checkpoint),
        },
        path,
    )
    print(f"saved checkpoint: {path}", flush=True)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint save failed: {path}")
    print(f"checkpoint exists: {path} ({path.stat().st_size / (1024 ** 2):.2f} MB)", flush=True)


def main():
    args = parse_args()
    root = args.root.resolve()
    checkpoint_path = resolve_path(root, args.checkpoint)
    output_dir = resolve_path(root, args.output_dir)
    require_file(checkpoint_path, "DreamDiffusion generation checkpoint")

    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    from dataset import create_EEG_dataset
    from dc_ldm.ldm_for_eeg import eLDM_eval

    device = torch.device(args.device)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but torch.cuda.is_available() is False.")
    require_file(checkpoint_path, "DreamDiffusion generation checkpoint")

    print(f"root: {root}", flush=True)
    print(f"device: {device}", flush=True)
    print(f"loading generation checkpoint: {checkpoint_path}", flush=True)

    checkpoint_payload = torch.load(checkpoint_path, map_location="cpu")
    if "model_state_dict" not in checkpoint_payload:
        raise KeyError(f"{checkpoint_path} does not contain key 'model_state_dict'.")

    config = prepare_config(args, checkpoint_payload)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    pretrain_root = Path(config.pretrain_gm_path)
    config_patch = resolve_path(root, args.config_patch) if args.config_patch else pretrain_root / "models/config15.yaml"
    require_file(config_patch, "Stable Diffusion config")
    require_file(config.eeg_signals_path, "EEG signals")
    require_file(config.splits_path, "dataset split")
    if args.imagenet_path is not None:
        require_file(resolve_path(root, args.imagenet_path), "ImageNet image directory")

    print(f"pretrain_root: {pretrain_root}", flush=True)
    print(f"eeg_signals_path: {config.eeg_signals_path}", flush=True)
    print(f"splits_path: {config.splits_path}", flush=True)
    print(f"imagenet_path: {args.imagenet_path}", flush=True)
    print(f"subject: {config.subject}", flush=True)
    print(f"batch_size: {config.batch_size}", flush=True)
    print(f"max_steps: {args.max_steps}", flush=True)
    print(f"lr: {config.lr}", flush=True)
    print(f"clip_tune: {config.clip_tune}", flush=True)
    print(f"cls_tune: {config.cls_tune}", flush=True)

    if config.dataset != "EEG":
        raise NotImplementedError("train_resume_sanity.py currently supports --dataset EEG only.")

    image_transform = make_train_transform(config)
    dataset_train, _ = create_EEG_dataset(
        eeg_signals_path=config.eeg_signals_path,
        splits_path=config.splits_path,
        imagenet_path=str(resolve_path(root, args.imagenet_path)) if args.imagenet_path else None,
        image_transform=[image_transform, image_transform],
        subject=config.subject,
    )
    print(f"train samples: {len(dataset_train)}", flush=True)
    num_voxels = dataset_train.data_len

    print(f"config_patch: {config_patch}", flush=True)

    print("building eLDM_eval from config, then loading generation checkpoint ...", flush=True)
    generative_model = eLDM_eval(
        str(config_patch),
        num_voxels,
        device=device,
        pretrain_root=str(pretrain_root),
        logger=None,
        ddim_steps=config.ddim_steps,
        global_pool=config.global_pool,
        use_time_cond=config.use_time_cond,
        clip_tune=config.clip_tune,
        cls_tune=config.cls_tune,
    )
    model = generative_model.model
    missing, unexpected = model.load_state_dict(checkpoint_payload["model_state_dict"], strict=False)
    print(f"loaded generation checkpoint with strict=False; missing={len(missing)}, unexpected={len(unexpected)}", flush=True)
    if missing:
        print(f"first missing keys: {missing[:10]}", flush=True)
    if unexpected:
        print(f"first unexpected keys: {unexpected[:10]}", flush=True)

    model.to(device)
    model.unfreeze_whole_model()
    model.freeze_first_stage()
    model.learning_rate = config.lr
    model.train_cond_stage_only = True
    model.eval_avg = getattr(config, "eval_avg", True)
    model.main_config = config
    model.output_path = str(output_dir)

    optimizer = build_optimizer(model)
    dataloader = DataLoader(
        dataset_train,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    if len(dataloader) == 0:
        raise RuntimeError("training dataloader is empty.")

    model.train()
    if model.cond_stage_model is not None:
        model.cond_stage_model.train()

    global_step = 0
    data_iter = iter(dataloader)
    while global_step < args.max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        batch = move_to_device(batch, device)
        if global_step == 0:
            check_batch_devices(batch, device)
        optimizer.zero_grad(set_to_none=True)
        loss, loss_dict = model.shared_step(batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        global_step += 1
        parts = [f"step {global_step}/{args.max_steps}", f"loss={tensor_to_float(loss):.6f}"]
        for key, value in sorted(loss_dict.items()):
            parts.append(f"{key}={tensor_to_float(value):.6f}")
        print(" | ".join(parts), flush=True)

        if args.save_every_n_steps and global_step % args.save_every_n_steps == 0:
            save_checkpoint(
                output_dir / f"resume_{global_step}steps.pth",
                model,
                config,
                optimizer,
                global_step,
                checkpoint_path,
                device,
            )

    final_path = output_dir / f"resume_{args.max_steps}steps.pth"
    save_checkpoint(final_path, model, config, optimizer, global_step, checkpoint_path, device)
    print(f"final checkpoint for gen_eval_eeg.py --model_path: {final_path}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", flush=True)
        raise SystemExit(1)
