import argparse
import copy
import csv
import json
import math
import types
import sys
from pathlib import Path

import torch


CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))


from dynamic_chunking_modules import (  # noqa: E402
    DiffusionTimeRouter,
    DynamicBoundaryPredictor,
    entropy,
    routing_regularization,
    soft_membership,
    token_prior,
)
from train_resume_sanity import (  # noqa: E402
    SkipBadImageDataset,
    check_batch_devices,
    collate_skip_none,
    make_train_transform,
    move_to_device,
    require_file,
    resolve_path,
    resolve_training_window,
    tensor_to_float,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Method-aligned Dynamic Chunk-Guided EEG Diffusion training."
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--dataset", type=str, default="EEG")
    parser.add_argument("--pretrained_generation_checkpoint", "--resume_from_checkpoint",
                        dest="checkpoint", type=Path, required=True)
    parser.add_argument("--eeg_signals_path", type=Path, default=Path("datasets/eeg_5_95_std.pth"))
    parser.add_argument("--splits_path", type=Path, default=Path("datasets/block_splits_by_image_single.pth"))
    parser.add_argument("--imagenet_path", type=Path, default=Path("datasets/imageNet_images"))
    parser.add_argument("--pretrain_root", type=Path, default=Path("pretrains"))
    parser.add_argument("--config_patch", type=Path, default=Path("pretrains/models/config15.yaml"))
    parser.add_argument("--output_dir", type=Path, default=Path("checkpoints/dynamic_chunking_method"))
    parser.add_argument("--max_steps", type=int, default=20)
    parser.add_argument("--start_step", type=int, default=None)
    parser.add_argument("--target_step", type=int, default=None)
    parser.add_argument("--save_every_n_steps", type=int, default=0)
    parser.add_argument("--log_every_n_steps", type=int, default=1)
    parser.add_argument("--save_routing_every_n_steps", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--subject", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_bad_images", action="store_true")
    parser.add_argument("--min_chunk_length", type=float, default=8.0)
    parser.add_argument("--boundary_hidden_dim", type=int, default=256)
    parser.add_argument("--boundary_temperature", type=float, default=1.0)
    parser.add_argument("--boundary_rho", type=float, default=1.0)
    parser.add_argument("--router_temperature", type=float, default=1.0)
    parser.add_argument("--router_weight_floor", type=float, default=0.03)
    parser.add_argument("--router_hidden_dim", type=int, default=256)
    parser.add_argument("--router_time_dim", type=int, default=64)
    parser.add_argument("--lambda_smooth", type=float, default=1e-3)
    parser.add_argument("--lambda_spec", type=float, default=1e-3)
    parser.add_argument("--routing_grid_size", type=int, default=250)
    parser.add_argument("--train_scope", choices=["method", "boundary_router_only", "router_only"], default="method")
    parser.add_argument("--sanity_check", action="store_true")
    parser.add_argument("--disable_clip_tune", action="store_true")
    parser.add_argument("--disable_cls_tune", action="store_true")
    args = parser.parse_args()

    if args.max_steps <= 0:
        parser.error("--max_steps must be positive.")
    if args.batch_size <= 0:
        parser.error("--batch_size must be positive.")
    if args.lr <= 0:
        parser.error("--lr must be positive.")
    if args.min_chunk_length <= 0:
        parser.error("--min_chunk_length must be positive.")
    if args.boundary_rho <= 0:
        parser.error("--boundary_rho must be positive.")
    if not (0 <= args.router_weight_floor < 1):
        parser.error("--router_weight_floor must be in [0, 1).")
    if args.lambda_smooth < 0 or args.lambda_spec < 0:
        parser.error("--lambda_smooth and --lambda_spec must be >= 0.")
    if args.routing_grid_size < 2:
        parser.error("--routing_grid_size must be >= 2.")
    return args


class MethodState:
    def __init__(self, boundary_predictor, router, temporal_projector, args):
        self.boundary_predictor = boundary_predictor
        self.router = router
        self.temporal_projector = temporal_projector
        self.args = args
        self.current_t = None
        self.num_timesteps = None
        self.h_tokens = None
        self.context = None
        self.lengths = None
        self.boundaries = None
        self.membership = None
        self.weights = None
        self.pi = None
        self.shape_audit = {}
        self.loss_key_audit_printed = False

    def start_batch(self, t, num_timesteps):
        self.current_t = t.detach()
        self.num_timesteps = int(num_timesteps)
        self.h_tokens = None
        self.context = None
        self.lengths = None
        self.boundaries = None
        self.membership = None
        self.weights = None
        self.pi = None

    def progress(self, batch_size, device, dtype):
        if self.current_t is None:
            raise RuntimeError("No diffusion timestep cached for dynamic method state.")
        denom = max(float(self.num_timesteps - 1), 1.0)
        p = (float(self.num_timesteps - 1) - self.current_t.to(device=device, dtype=dtype)) / denom
        if p.shape[0] != batch_size:
            if p.shape[0] == 1:
                p = p.expand(batch_size)
            else:
                raise RuntimeError(f"progress batch {p.shape[0]} != context batch {batch_size}")
        return p

    def encode_condition(self, cond_stage, x):
        h = cond_stage.mae(x)
        self.h_tokens = h
        c = self.temporal_projector(h)
        self.context = c
        lengths, boundaries, _ = self.boundary_predictor(h)
        membership = soft_membership(boundaries, h.shape[1], rho=self.args.boundary_rho)
        self.lengths = lengths
        self.boundaries = boundaries
        self.membership = membership
        self.shape_audit = {
            "raw_eeg_X": list(x.shape),
            "encoder_H": list(h.shape),
            "temporal_context_C": list(c.shape),
            "N_H": int(h.shape[1]),
            "N_C": int(c.shape[1]),
            "H_feature_dim": int(h.shape[-1]),
            "C_feature_dim": int(c.shape[-1]),
            "projector_feature_only": True,
            "projector_mixes_token_dimension": False,
            "existing_conditioner_has_channel_mapper": hasattr(cond_stage, "channel_mapper"),
            "existing_channel_mapper_note": (
                "The original/current conditioner may pool token dimension to 77; "
                "this method path bypasses it and uses temporal_projector(H) with N_C=N_H."
            ),
        }
        return c, h

    def bias_for_context(self, context, dtype):
        if self.membership is None:
            raise RuntimeError("Soft membership is not initialized. Did cond_stage.forward run?")
        if context.shape[1] != self.membership.shape[-1]:
            raise RuntimeError(
                f"context tokens {context.shape[1]} != membership tokens {self.membership.shape[-1]}"
            )
        progress = self.progress(context.shape[0], context.device, context.dtype)
        weights, _ = self.router(progress)
        if weights.shape != (context.shape[0], 3):
            raise RuntimeError(f"router weights shape {list(weights.shape)} != {[context.shape[0], 3]}")
        pi = token_prior(weights, self.membership.to(device=context.device, dtype=weights.dtype))
        self.weights = weights
        self.pi = pi
        return torch.log(pi + 1e-8).to(dtype=dtype)

    def sampled_stats(self):
        stats = {}
        if self.weights is not None:
            w = self.weights.detach().float()
            ent = entropy(w).mean()
            stats.update({
                "dynamic/w1": float(w[:, 0].mean().cpu()),
                "dynamic/w2": float(w[:, 1].mean().cpu()),
                "dynamic/w3": float(w[:, 2].mean().cpu()),
                "dynamic/router_entropy": float(ent.cpu()),
            })
        if self.boundaries is not None:
            b = self.boundaries.detach().float()
            l = self.lengths.detach().float()
            stats.update({
                "dynamic/b1_mean": float(b[:, 1].mean().cpu()),
                "dynamic/b1_std": float(b[:, 1].std(unbiased=False).cpu()),
                "dynamic/b2_mean": float(b[:, 2].mean().cpu()),
                "dynamic/b2_std": float(b[:, 2].std(unbiased=False).cpu()),
                "dynamic/l1_mean": float(l[:, 0].mean().cpu()),
                "dynamic/l2_mean": float(l[:, 1].mean().cpu()),
                "dynamic/l3_mean": float(l[:, 2].mean().cpu()),
                "dynamic/min_length": float(l.min().cpu()),
                "dynamic/max_length": float(l.max().cpu()),
            })
        return stats


def prepare_config(args, checkpoint_payload):
    from config import Config_Generative_Model

    config = copy.deepcopy(checkpoint_payload["config"]) if "config" in checkpoint_payload else Config_Generative_Model()
    root = args.root.resolve()
    config.root_path = str(root)
    config.output_path = str(resolve_path(root, args.output_dir))
    config.dataset = args.dataset
    config.logger = None
    config.trainer = None
    config.eeg_signals_path = str(resolve_path(root, args.eeg_signals_path))
    config.splits_path = str(resolve_path(root, args.splits_path))
    config.pretrain_gm_path = str(resolve_path(root, args.pretrain_root))
    config.batch_size = args.batch_size
    config.lr = args.lr
    config.subject = args.subject
    config.seed = args.seed
    config.clip_tune = False if args.disable_clip_tune else getattr(config, "clip_tune", True)
    config.cls_tune = False if args.disable_cls_tune else getattr(config, "cls_tune", False)
    for key, value in {
        "eval_avg": True,
        "global_pool": False,
        "use_time_cond": True,
        "crop_ratio": 0.2,
        "img_size": 512,
        "ddim_steps": 250,
        "HW": None,
    }.items():
        if not hasattr(config, key):
            setattr(config, key, value)
    config.dynamic_chunking_method = {
        "enabled": True,
        "min_chunk_length": args.min_chunk_length,
        "boundary_hidden_dim": args.boundary_hidden_dim,
        "boundary_temperature": args.boundary_temperature,
        "boundary_rho": args.boundary_rho,
        "router_temperature": args.router_temperature,
        "router_weight_floor": args.router_weight_floor,
        "router_hidden_dim": args.router_hidden_dim,
        "router_time_dim": args.router_time_dim,
        "lambda_smooth": args.lambda_smooth,
        "lambda_spec": args.lambda_spec,
        "train_scope": args.train_scope,
        "uses_fixed_boundaries": False,
        "uses_moe": False,
    }
    return config


def patch_conditioner_forward(cond_stage, method_state):
    original_forward = cond_stage.forward

    def temporal_forward(self, x):
        return method_state.encode_condition(self, x)

    cond_stage.forward = types.MethodType(temporal_forward, cond_stage)
    return original_forward


def patch_p_losses(model, method_state):
    original = model.p_losses

    def patched(x_start, cond, t, noise=None):
        method_state.start_batch(t, model.num_timesteps)
        return original(x_start, cond, t, noise=noise)

    model.p_losses = patched
    return original


def patch_attn2_modules(model, method_state):
    from einops import rearrange, repeat
    from dc_ldm.modules.attention import CrossAttention, default, exists

    patched = []
    for name, module in model.named_modules():
        if not name.endswith("attn2") or not isinstance(module, CrossAttention):
            continue
        original = module.forward

        def make_forward(attn_module, module_name, original_forward):
            def forward(x, context=None, mask=None):
                if context is None:
                    return original_forward(x, context=context, mask=mask)
                h = attn_module.heads
                q = attn_module.to_q(x)
                context_value = default(context, x)
                k = attn_module.to_k(context_value)
                v = attn_module.to_v(context_value)
                q, k, v = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v))
                sim = torch.einsum("b i d, b j d -> b i j", q, k) * attn_module.scale
                if context is not x:
                    bias_value = method_state.bias_for_context(context_value, sim.dtype)
                    if sim.shape[-1] != bias_value.shape[-1]:
                        raise RuntimeError(
                            f"{module_name}: attention context length {sim.shape[-1]} != bias length {bias_value.shape[-1]}"
                        )
                    bias = repeat(bias_value, "b n -> (b h) 1 n", h=h)
                    if sim.shape[0] == bias.shape[0]:
                        sim = sim + bias
                    elif sim.shape[0] == bias.shape[0] * 2:
                        full = torch.zeros_like(sim[:, :1, :])
                        full[bias.shape[0]:] = bias
                        sim = sim + full
                    else:
                        raise RuntimeError(
                            f"{module_name}: cannot broadcast bias batch-heads {bias.shape[0]} to {sim.shape[0]}"
                        )
                if exists(mask):
                    mask_r = rearrange(mask, "b ... -> b (...)")
                    max_neg_value = -torch.finfo(sim.dtype).max
                    mask_r = repeat(mask_r, "b j -> (b h) () j", h=h)
                    sim.masked_fill_(~mask_r, max_neg_value)
                attn = sim.softmax(dim=-1)
                out = torch.einsum("b i j, b j d -> b i d", attn, v)
                out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
                return attn_module.to_out(out)
            return forward

        module.forward = make_forward(module, name, original)
        patched.append((module, original, name))
    if not patched:
        raise RuntimeError("No attn2 CrossAttention modules were found to patch.")
    return patched


def restore_attn2_modules(patched):
    for module, original, _ in patched:
        module.forward = original


def initialize_temporal_projector(cond_stage, out_dim=768):
    in_dim = cond_stage.mae.embed_dim
    projector = torch.nn.Linear(in_dim, out_dim, bias=True)
    if hasattr(cond_stage, "dim_mapper") and tuple(cond_stage.dim_mapper.weight.shape) == tuple(projector.weight.shape):
        projector.load_state_dict(cond_stage.dim_mapper.state_dict())
        init_note = "initialized from existing cond_stage_model.dim_mapper"
    else:
        torch.nn.init.xavier_uniform_(projector.weight)
        torch.nn.init.zeros_(projector.bias)
        init_note = "xavier_uniform because existing dim_mapper shape did not match"
    return projector, init_note


def allowed_missing(key):
    prefixes = (
        "dynamic_boundary_predictor.",
        "diffusion_time_router.",
        "dynamic_temporal_projector.",
    )
    return key.startswith(prefixes)


def allowed_unexpected(key):
    prefixes = (
        "dynamic_boundary_predictor.",
        "diffusion_time_router.",
        "dynamic_temporal_projector.",
        "dynamic_chunk_router.",
        "cond_stage_model.channel_mapper.",
    )
    exact = {"image_embedder.transformer.vision_model.embeddings.position_ids"}
    return key.startswith(prefixes) or key in exact


def validate_load_keys(missing, unexpected):
    bad_missing = [key for key in missing if not allowed_missing(key)]
    bad_unexpected = [key for key in unexpected if not allowed_unexpected(key)]
    if bad_missing:
        raise RuntimeError(f"Unexpected missing checkpoint keys: {bad_missing[:30]}")
    if bad_unexpected:
        raise RuntimeError(f"Unexpected checkpoint keys: {bad_unexpected[:30]}")


def configure_trainable(model, scope):
    for param in model.parameters():
        param.requires_grad = False
    trainable_prefixes = ["dynamic_boundary_predictor.", "diffusion_time_router.", "dynamic_temporal_projector."]
    if scope == "method":
        trainable_prefixes.extend(["cond_stage_model.mae.", "cond_stage_model.mapping."])
        trainable_contains = ["attn2"]
    elif scope == "boundary_router_only":
        trainable_contains = []
    elif scope == "router_only":
        trainable_prefixes = ["diffusion_time_router."]
        trainable_contains = []
    else:
        raise ValueError(scope)

    for name, param in model.named_parameters():
        if any(name.startswith(prefix) for prefix in trainable_prefixes) or any(token in name for token in trainable_contains):
            param.requires_grad = True

    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable parameters selected.")
    return trainable


def grad_norm_for(model, predicate):
    total = 0.0
    found = False
    for name, param in model.named_parameters():
        if predicate(name) and param.grad is not None:
            found = True
            total += float(param.grad.detach().float().norm().cpu()) ** 2
    return math.sqrt(total) if found else 0.0


def print_trainable_report(model):
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    total = sum(p.numel() for _, p in trainable)
    print(f"trainable parameter tensors: {len(trainable)}", flush=True)
    print(f"trainable parameter count: {total}", flush=True)
    for name, param in trainable:
        print(f"  trainable {name} shape={list(param.shape)} numel={param.numel()}", flush=True)


def freeze_runtime_modes(model):
    if hasattr(model, "first_stage_model") and model.first_stage_model is not None:
        model.first_stage_model.eval()
    if hasattr(model, "image_embedder") and model.image_embedder is not None:
        model.image_embedder.eval()


def save_routing_curve(path, router, grid_size, device):
    path.parent.mkdir(parents=True, exist_ok=True)
    progress = torch.linspace(0.0, 1.0, grid_size, device=device)
    with torch.no_grad():
        weights, _ = router(progress)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["p", "w1", "w2", "w3"])
        writer.writeheader()
        for p_value, w in zip(progress.detach().cpu().tolist(), weights.detach().cpu().tolist()):
            writer.writerow({"p": p_value, "w1": w[0], "w2": w[1], "w3": w[2]})


def save_boundary_stats(path, method_state):
    path.parent.mkdir(parents=True, exist_ok=True)
    stats = method_state.sampled_stats()
    stats["shape_audit"] = method_state.shape_audit
    path.write_text(json.dumps(stats, indent=2))


def save_checkpoint(path, model, optimizer, config, args, method_state, run_step, total_step, start_step, target_step, source, device):
    path.parent.mkdir(parents=True, exist_ok=True)
    cfg = copy.deepcopy(config)
    cfg.logger = None
    cfg.trainer = None
    rng_state = torch.cuda.get_rng_state(device) if device.type == "cuda" else torch.random.get_rng_state()
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "global_step": total_step,
        "total_step": total_step,
        "run_step": run_step,
        "start_step": start_step,
        "target_step": target_step,
        "config": cfg,
        "state": rng_state,
        "source_checkpoint": str(source),
        "dynamic_chunking_method": {
            "boundary_predictor": {
                "min_chunk_length": args.min_chunk_length,
                "hidden_dim": args.boundary_hidden_dim,
                "temperature": args.boundary_temperature,
            },
            "router": {
                "temperature": args.router_temperature,
                "weight_floor": args.router_weight_floor,
                "hidden_dim": args.router_hidden_dim,
                "time_dim": args.router_time_dim,
            },
            "rho": args.boundary_rho,
            "lambda_smooth": args.lambda_smooth,
            "lambda_spec": args.lambda_spec,
            "diagnostics": method_state.sampled_stats(),
        },
    }, path)
    print(f"saved checkpoint: {path}", flush=True)


def build_model_and_data(args):
    import numpy as np
    from torch.utils.data import DataLoader
    from dataset import create_EEG_dataset
    from dc_ldm.ldm_for_eeg import eLDM_eval

    root = args.root.resolve()
    checkpoint_path = resolve_path(root, args.checkpoint)
    for path, desc in [
        (checkpoint_path, "checkpoint"),
        (resolve_path(root, args.eeg_signals_path), "EEG signals"),
        (resolve_path(root, args.splits_path), "split file"),
        (resolve_path(root, args.imagenet_path), "ImageNet path"),
        (resolve_path(root, args.config_patch), "config"),
    ]:
        require_file(path, desc)
    payload = torch.load(checkpoint_path, map_location="cpu")
    config = prepare_config(args, payload)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA unavailable.")

    transform = make_train_transform(config)
    dataset_train, _ = create_EEG_dataset(
        eeg_signals_path=config.eeg_signals_path,
        splits_path=config.splits_path,
        imagenet_path=str(resolve_path(root, args.imagenet_path)),
        image_transform=[transform, transform],
        subject=config.subject,
    )
    model_wrap = eLDM_eval(
        str(resolve_path(root, args.config_patch)),
        dataset_train.data_len,
        device=device,
        pretrain_root=str(resolve_path(root, args.pretrain_root)),
        logger=None,
        ddim_steps=config.ddim_steps,
        global_pool=config.global_pool,
        use_time_cond=config.use_time_cond,
        clip_tune=config.clip_tune,
        cls_tune=config.cls_tune,
    )
    model = model_wrap.model
    temporal_projector, init_note = initialize_temporal_projector(model.cond_stage_model, out_dim=768)
    model.dynamic_temporal_projector = temporal_projector
    model.dynamic_boundary_predictor = DynamicBoundaryPredictor(
        input_dim=model.cond_stage_model.mae.embed_dim,
        hidden_dim=args.boundary_hidden_dim,
        min_chunk_length=args.min_chunk_length,
        temperature=args.boundary_temperature,
    )
    model.diffusion_time_router = DiffusionTimeRouter(
        time_dim=args.router_time_dim,
        hidden_dim=args.router_hidden_dim,
        temperature=args.router_temperature,
        weight_floor=args.router_weight_floor,
    )
    missing, unexpected = model.load_state_dict(payload["model_state_dict"], strict=False)
    validate_load_keys(missing, unexpected)
    print(f"load_state_dict strict=False missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    if missing:
        print(f"allowed missing keys: {missing[:20]}", flush=True)
    if unexpected:
        print(f"allowed unexpected keys: {unexpected[:20]}", flush=True)
    print(f"temporal projector init: {init_note}", flush=True)

    model.to(device)
    model.unfreeze_whole_model()
    model.freeze_first_stage()
    model.learning_rate = config.lr
    model.train_cond_stage_only = True
    model.main_config = config
    model.output_path = str(resolve_path(root, args.output_dir))
    method_state = MethodState(model.dynamic_boundary_predictor, model.diffusion_time_router, model.dynamic_temporal_projector, args)
    original_cond_forward = patch_conditioner_forward(model.cond_stage_model, method_state)
    original_p_losses = patch_p_losses(model, method_state)
    patched_attn = patch_attn2_modules(model, method_state)
    trainable = configure_trainable(model, args.train_scope)
    optimizer = torch.optim.AdamW([p for _, p in trainable], lr=args.lr)
    if "optimizer_state_dict" in payload and "dynamic_chunking_method" in payload:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        print("restored optimizer_state_dict from dynamic checkpoint", flush=True)
    if "state" in payload and "dynamic_chunking_method" in payload:
        try:
            if device.type == "cuda":
                torch.cuda.set_rng_state(payload["state"].cpu(), device)
            else:
                torch.random.set_rng_state(payload["state"])
            print("restored RNG state from dynamic checkpoint", flush=True)
        except Exception as exc:
            print(f"[WARN] could not restore RNG state: {type(exc).__name__}: {exc}", flush=True)

    dataset_for_loader = SkipBadImageDataset(dataset_train) if args.skip_bad_images else dataset_train
    dataloader_kwargs = {"collate_fn": collate_skip_none} if args.skip_bad_images else {}
    dataloader = DataLoader(
        dataset_for_loader,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        **dataloader_kwargs,
    )
    patches = (original_cond_forward, original_p_losses, patched_attn)
    return model, optimizer, dataloader, config, payload, method_state, patches, device, checkpoint_path


def restore_patches(model, patches):
    original_cond_forward, original_p_losses, patched_attn = patches
    model.cond_stage_model.forward = original_cond_forward
    model.p_losses = original_p_losses
    restore_attn2_modules(patched_attn)


def loss_audit(base_loss, loss_dict):
    align = loss_dict.get("train/loss_clip", loss_dict.get("val/loss_clip", torch.tensor(0.0, device=base_loss.device)))
    cls = loss_dict.get("train/loss_cls", loss_dict.get("val/loss_cls", torch.tensor(0.0, device=base_loss.device)))
    l_diff = base_loss - align - cls
    return l_diff, align


def run_one_backward(model, batch, method_state, args, device, optimizer=None):
    batch = move_to_device(batch, device)
    check_batch_devices(batch, device)
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
    base_loss, loss_dict = model.shared_step(batch)
    l_diff, l_align = loss_audit(base_loss, loss_dict)
    reg_loss, reg_terms, _ = routing_regularization(
        model.diffusion_time_router,
        grid_size=args.routing_grid_size,
        lambda_smooth=args.lambda_smooth,
        lambda_spec=args.lambda_spec,
        device=device,
    )
    total_loss = base_loss + reg_loss
    total_loss.backward()
    return total_loss, base_loss, l_diff, l_align, loss_dict, reg_terms


def assert_sanity(model, method_state, args, device):
    b, n, d = 4, 110, model.cond_stage_model.mae.embed_dim
    h = torch.randn(b, n, d, device=device, requires_grad=True)
    lengths, boundaries, _ = model.dynamic_boundary_predictor(h)
    if lengths.shape != (b, 3):
        raise AssertionError(f"lengths shape {list(lengths.shape)}")
    if (lengths < args.min_chunk_length - 1e-4).any():
        raise AssertionError("length below min_chunk_length")
    if not torch.allclose(lengths.sum(dim=-1), torch.full((b,), float(n), device=device), atol=1e-4):
        raise AssertionError("chunk lengths do not sum to N")
    if not ((boundaries[:, 0] == 0).all() and (boundaries[:, 1] > 0).all() and
            (boundaries[:, 1] < boundaries[:, 2]).all() and (boundaries[:, 2] < n).all()):
        raise AssertionError("invalid boundaries")
    membership = soft_membership(boundaries, n, rho=args.boundary_rho)
    if membership.shape != (b, 3, n):
        raise AssertionError(f"membership shape {list(membership.shape)}")
    if not torch.allclose(membership.sum(dim=1), torch.ones(b, n, device=device), atol=1e-4):
        raise AssertionError("membership does not sum to 1 over chunks")
    membership_objective = (
        membership
        * torch.arange(n, device=device, dtype=membership.dtype).view(1, 1, n)
        * torch.tensor([1.0, 2.0, 3.0], device=device, dtype=membership.dtype).view(1, 3, 1)
    ).mean()
    membership_objective.backward(retain_graph=True)
    if h.grad is None or h.grad.abs().sum() == 0:
        raise AssertionError("membership gradient does not reach boundary predictor input")
    model.zero_grad(set_to_none=True)

    for p in [0.0, 1.0]:
        progress = torch.full((b,), p, device=device)
        weights, _ = model.diffusion_time_router(progress)
        if not (weights > 0).all():
            raise AssertionError("router weights are not positive")
        if not torch.allclose(weights.sum(dim=-1), torch.ones(b, device=device), atol=1e-6):
            raise AssertionError("router weights do not sum to 1")
        if not torch.allclose(weights[0], weights[1], atol=1e-7):
            raise AssertionError("same progress should produce same weights across samples")
    pi = token_prior(weights, membership)
    if pi.shape != (b, n) or not (pi > 0).all():
        raise AssertionError("invalid token prior")

    q = torch.randn(2, 5, 16, device=device)
    k = torch.randn(2, n, 16, device=device)
    v = torch.randn(2, n, 16, device=device)
    sim = torch.einsum("bid,bjd->bij", q, k) / math.sqrt(16)
    baseline = sim.softmax(dim=-1) @ v
    uniform_bias = math.log(1.0 / 3.0)
    routed = (sim + uniform_bias).softmax(dim=-1) @ v
    max_diff = (baseline - routed).abs().max().item()
    if max_diff > 1e-5:
        raise AssertionError(f"uniform baseline equivalence failed: {max_diff}")
    print("sanity module tests passed", flush=True)


def main():
    args = parse_args()
    model, optimizer, dataloader, config, payload, method_state, patches, device, source = build_model_and_data(args)
    output_dir = resolve_path(args.root.resolve(), args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    start_step, target_step, run_steps, inferred_step, inferred_from = resolve_training_window(args, payload)

    print(f"checkpoint inferred step: {inferred_step} from {inferred_from}", flush=True)
    print(f"start_step: {start_step}", flush=True)
    print(f"target_step: {target_step}", flush=True)
    print(f"run_steps: {run_steps}", flush=True)
    print(f"train_scope: {args.train_scope}", flush=True)
    print_trainable_report(model)
    model.train()
    freeze_runtime_modes(model)

    try:
        if args.sanity_check:
            assert_sanity(model, method_state, args, device)
        data_iter = iter(dataloader)
        run_step = 0
        planned_steps = 2 if args.sanity_check else run_steps
        while run_step < planned_steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)
            if batch is None:
                continue
            do_step = True
            total_loss, base_loss, l_diff, l_align, loss_dict, reg_terms = run_one_backward(
                model, batch, method_state, args, device, optimizer=optimizer
            )
            if not method_state.loss_key_audit_printed:
                print(f"loss_dict actual keys: {sorted(loss_dict.keys())}", flush=True)
                print("L_diff is reported as base shared_step loss minus train/loss_clip and train/loss_cls when present.", flush=True)
                print("L_align is train/loss_clip when present; CLIP alignment uses global H returned by cond_stage_model.forward.", flush=True)
                print(f"shape audit: {json.dumps(method_state.shape_audit, indent=2)}", flush=True)
                method_state.loss_key_audit_printed = True

            total_step = start_step + run_step + 1
            if do_step:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 0.5)
                optimizer.step()
            stats = method_state.sampled_stats()
            grad_stats = {
                "grad/boundary": grad_norm_for(model, lambda n: n.startswith("dynamic_boundary_predictor.")),
                "grad/router": grad_norm_for(model, lambda n: n.startswith("diffusion_time_router.")),
                "grad/eeg_encoder": grad_norm_for(model, lambda n: n.startswith("cond_stage_model.mae.")),
                "grad/eeg_projector": grad_norm_for(model, lambda n: n.startswith("dynamic_temporal_projector.")),
                "grad/cross_attention": grad_norm_for(model, lambda n: "attn2" in n),
            }
            parts = [
                f"run_step {run_step + 1}/{planned_steps if args.sanity_check else run_steps}",
                f"total_step {total_step}/{target_step}",
                f"loss={tensor_to_float(total_loss):.6f}",
                f"L_diff={tensor_to_float(l_diff):.6f}",
                f"L_align={tensor_to_float(l_align):.6f}",
            ]
            for key, val in sorted(reg_terms.items()):
                parts.append(f"{key}={tensor_to_float(val):.6f}")
            for key, val in sorted(stats.items()):
                parts.append(f"{key}={float(val):.6f}")
            for key, val in sorted(grad_stats.items()):
                parts.append(f"{key}={float(val):.6f}")
            print(" | ".join(parts), flush=True)

            if args.sanity_check:
                if run_step == 0:
                    print("sanity warmup optimizer step completed; checking full gradients on the next batch.", flush=True)
                    run_step += 1
                    continue
                frozen_bad = [
                    name for name, param in model.named_parameters()
                    if (not param.requires_grad) and param.grad is not None and param.grad.abs().sum() > 0
                ]
                if frozen_bad:
                    raise AssertionError(f"Frozen parameters received gradients: {frozen_bad[:20]}")
                for label, value in grad_stats.items():
                    if args.train_scope == "method" and value == 0:
                        raise AssertionError(f"Expected nonzero gradient for {label}")
                print("sanity backward/frozen tests passed", flush=True)
                break

            run_step += 1
            if args.save_routing_every_n_steps and run_step % args.save_routing_every_n_steps == 0:
                save_routing_curve(output_dir / f"routing_curve_step_{total_step:06d}.csv",
                                   model.diffusion_time_router, args.routing_grid_size, device)
                save_boundary_stats(output_dir / f"boundary_stats_step_{total_step:06d}.json", method_state)
            if args.save_every_n_steps and run_step % args.save_every_n_steps == 0:
                save_checkpoint(output_dir / f"dynamic_method_total_{total_step}steps.pth", model, optimizer,
                                config, args, method_state, run_step, total_step, start_step, target_step, source, device)

        if not args.sanity_check:
            save_routing_curve(output_dir / f"routing_curve_step_{target_step:06d}.csv",
                               model.diffusion_time_router, args.routing_grid_size, device)
            save_boundary_stats(output_dir / f"boundary_stats_step_{target_step:06d}.json", method_state)
            save_checkpoint(output_dir / f"dynamic_method_total_{target_step}steps.pth", model, optimizer,
                            config, args, method_state, run_steps, target_step, start_step, target_step, source, device)
    finally:
        restore_patches(model, patches)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", flush=True)
        raise SystemExit(1)
