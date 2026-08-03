import argparse
import copy
import math
import sys
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))


from run_fixed_chunking import chunk_boundaries, membership_matrix, validate_chunks  # noqa: E402
from train_resume_sanity import (  # noqa: E402
    SkipBadImageDataset,
    channel_last,
    check_batch_devices,
    collate_skip_none,
    infer_checkpoint_step,
    make_train_transform,
    move_to_device,
    normalize,
    random_crop,
    require_file,
    resolve_path,
    resolve_training_window,
    tensor_to_float,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train a dynamic EEG chunk router on top of an existing DreamDiffusion "
            "generation checkpoint. Chunk boundaries match code/run_fixed_chunking.py."
        )
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
    parser.add_argument("--output_dir", type=Path, default=Path("checkpoints/dynamic_chunking"))
    parser.add_argument("--max_steps", type=int, default=20)
    parser.add_argument("--start_step", type=int, default=None)
    parser.add_argument("--target_step", type=int, default=None)
    parser.add_argument("--save_every_n_steps", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--router_lr", type=float, default=None,
                        help="Router learning rate. Defaults to --lr.")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--subject", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_bad_images", action="store_true")
    parser.add_argument("--router_hidden_dim", type=int, default=256)
    parser.add_argument("--router_time_dim", type=int, default=64)
    parser.add_argument("--router_init_alpha", type=float, default=1e-4,
                        help="Initial residual bias strength. Small values start near baseline.")
    parser.add_argument("--router_max_alpha", type=float, default=1.0)
    parser.add_argument("--entropy_weight", type=float, default=0.0,
                        help="Optional penalty log(3)-H(w) to discourage early chunk collapse.")
    parser.add_argument("--alpha_l2_weight", type=float, default=0.0)
    parser.add_argument(
        "--train_scope",
        choices=["router_only", "router_conditioner", "router_crossattn_conditioner"],
        default="router_only",
        help=(
            "router_only isolates dynamic routing. router_conditioner also tunes EEG conditioner. "
            "router_crossattn_conditioner also tunes U-Net cross-attention/norm2/time condition params."
        ),
    )
    parser.add_argument("--disable_clip_tune", action="store_true")
    parser.add_argument("--disable_cls_tune", action="store_true")
    args = parser.parse_args()

    if args.max_steps <= 0:
        parser.error("--max_steps must be positive.")
    if args.batch_size <= 0:
        parser.error("--batch_size must be positive.")
    if args.lr <= 0:
        parser.error("--lr must be positive.")
    if args.router_lr is not None and args.router_lr <= 0:
        parser.error("--router_lr must be positive.")
    if args.router_init_alpha < 0:
        parser.error("--router_init_alpha must be >= 0.")
    if args.router_max_alpha <= 0:
        parser.error("--router_max_alpha must be positive.")
    if args.entropy_weight < 0 or args.alpha_l2_weight < 0:
        parser.error("--entropy_weight and --alpha_l2_weight must be >= 0.")
    return args


def softplus_inverse(value):
    import math

    value = max(float(value), 1e-8)
    return math.log(math.expm1(value))


def build_dynamic_chunk_router(context_dim, hidden_dim, time_dim, init_alpha, max_alpha):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class _DynamicChunkRouter(nn.Module):
        def __init__(self):
            super().__init__()
            self.time_dim = int(time_dim)
            self.max_alpha = float(max_alpha)
            self.net = nn.Sequential(
                nn.Linear(context_dim + time_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 3),
            )
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)
            self.raw_alpha = nn.Parameter(torch.tensor(softplus_inverse(init_alpha), dtype=torch.float32))

        def time_embedding(self, progress):
            half = self.time_dim // 2
            if half <= 0:
                return progress[:, None]
            freqs = torch.exp(
                -math.log(10000.0)
                * torch.arange(half, device=progress.device, dtype=progress.dtype)
                / max(half - 1, 1)
            )
            args = progress[:, None] * freqs[None, :]
            emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
            if emb.shape[-1] < self.time_dim:
                emb = F.pad(emb, (0, self.time_dim - emb.shape[-1]))
            return emb

        def alpha(self):
            return torch.clamp(F.softplus(self.raw_alpha), max=self.max_alpha)

        def forward(self, context, denoise_progress):
            pooled = context.mean(dim=1)
            if denoise_progress.ndim == 0:
                denoise_progress = denoise_progress.expand(context.shape[0])
            denoise_progress = denoise_progress.to(device=context.device, dtype=context.dtype).reshape(-1)
            if denoise_progress.shape[0] != context.shape[0]:
                if denoise_progress.shape[0] == 1:
                    denoise_progress = denoise_progress.expand(context.shape[0])
                else:
                    raise RuntimeError(
                        f"Router timestep batch {denoise_progress.shape[0]} does not match context batch {context.shape[0]}."
                    )
            emb = self.time_embedding(denoise_progress)
            logits = self.net(torch.cat([pooled, emb], dim=-1))
            return torch.softmax(logits, dim=-1)

    return _DynamicChunkRouter()


class DynamicRoutingState:
    def __init__(self, router, eps=1e-8):
        self.router = router
        self.eps = eps
        self.bounds = None
        self.membership = None
        self.token_count = None
        self.current_t = None
        self.num_timesteps = None
        self.cached_context_shape = None
        self.cached_weights = None
        self.last_weights = None
        self.last_entropy = None
        self.last_alpha = None
        self.debug_shapes = {}

    def set_token_count(self, token_count, device):
        self.token_count = int(token_count)
        self.bounds = chunk_boundaries(self.token_count, 3)
        validate_chunks(self.token_count, self.bounds)
        self.membership = membership_matrix(self.token_count, self.bounds, device=device)

    def start_batch(self, t, num_timesteps):
        self.current_t = t.detach()
        self.num_timesteps = int(num_timesteps)
        self.cached_context_shape = None
        self.cached_weights = None
        self.last_weights = None
        self.last_entropy = None
        self.last_alpha = None

    def denoise_progress(self, context_batch):
        import torch

        if self.current_t is None:
            raise RuntimeError("Dynamic router has no current diffusion timestep. Did p_losses patch run?")
        denom = max(float(self.num_timesteps - 1), 1.0)
        progress = 1.0 - self.current_t.to(dtype=torch.float32, device=self.current_t.device) / denom
        if progress.shape[0] != context_batch:
            if progress.shape[0] == 1:
                progress = progress.expand(context_batch)
            else:
                raise RuntimeError(
                    f"Diffusion timestep batch {progress.shape[0]} does not match context batch {context_batch}."
                )
        return progress

    def weights_for_context(self, context):
        if self.token_count is None:
            self.set_token_count(context.shape[1], context.device)
        if context.shape[1] != self.token_count:
            raise RuntimeError(f"Context token count changed from {self.token_count} to {context.shape[1]}.")

        context_shape = tuple(context.shape)
        if self.cached_weights is not None and self.cached_context_shape == context_shape:
            return self.cached_weights

        progress = self.denoise_progress(context.shape[0]).to(device=context.device)
        weights = self.router(context, progress)
        if weights.shape != (context.shape[0], 3):
            raise RuntimeError(f"Dynamic router returned shape {list(weights.shape)}, expected {[context.shape[0], 3]}.")
        if (weights <= 0).any():
            raise RuntimeError("Dynamic router produced non-positive weights.")
        entropy = -(weights * torch.log(weights + self.eps)).sum(dim=-1)

        self.cached_context_shape = context_shape
        self.cached_weights = weights
        self.last_weights = weights
        self.last_entropy = entropy
        self.last_alpha = self.router.alpha()
        self.debug_shapes.setdefault("cross_attention_context_shape", list(context.shape))
        self.debug_shapes.setdefault("chunk_boundaries", self.bounds)
        return weights

    def bias_for_context(self, context, dtype):
        import torch

        weights = self.weights_for_context(context)
        membership = self.membership.to(device=context.device, dtype=weights.dtype)
        pi = weights @ membership
        if (pi <= 0).any():
            raise RuntimeError(f"Token prior pi has non-positive values: min={pi.min().item()}")
        bias = torch.log(pi + self.eps) * self.router.alpha().to(device=context.device, dtype=weights.dtype)
        return bias.to(dtype=dtype)

    def regularization_loss(self, entropy_weight, alpha_l2_weight):
        import torch

        loss = torch.tensor(0.0, device=next(self.router.parameters()).device)
        terms = {}
        if self.last_entropy is not None and entropy_weight:
            entropy_penalty = math.log(3.0) - self.last_entropy.mean()
            loss = loss + float(entropy_weight) * entropy_penalty
            terms["dynamic/entropy_penalty"] = entropy_penalty
            terms["dynamic/entropy"] = self.last_entropy.mean()
        if alpha_l2_weight:
            alpha = self.router.alpha()
            alpha_l2 = alpha * alpha
            loss = loss + float(alpha_l2_weight) * alpha_l2
            terms["dynamic/alpha_l2"] = alpha_l2
        return loss, terms

    def stats(self):
        if self.last_weights is None:
            return {}
        weights = self.last_weights.detach().float().mean(dim=0).cpu().tolist()
        entropy = float(self.last_entropy.detach().float().mean().cpu()) if self.last_entropy is not None else None
        alpha = float(self.router.alpha().detach().float().cpu())
        return {
            "dynamic/w1": weights[0],
            "dynamic/w2": weights[1],
            "dynamic/w3": weights[2],
            "dynamic/entropy": entropy,
            "dynamic/alpha": alpha,
        }


def patch_cross_attention(dynamic_state):
    import contextlib
    import torch
    from einops import rearrange, repeat
    from dc_ldm.modules.attention import CrossAttention, default, exists

    @contextlib.contextmanager
    def _patched():
        original_forward = CrossAttention.forward

        def dynamic_forward(self, x, context=None, mask=None):
            h = self.heads
            q = self.to_q(x)
            context = default(context, x)
            is_cross = context is not x and context is not None
            raw_context = context
            k = self.to_k(context)
            v = self.to_v(context)
            q, k, v = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v))
            sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale

            if is_cross:
                bias_value = dynamic_state.bias_for_context(raw_context, sim.dtype)
                if sim.shape[-1] != bias_value.shape[-1]:
                    raise RuntimeError(
                        f"Attention context length {sim.shape[-1]} does not match dynamic bias length {bias_value.shape[-1]}."
                    )
                if sim.shape[0] == bias_value.shape[0] * h:
                    bias = repeat(bias_value, "b n -> (b h) 1 n", h=h)
                    sim = sim + bias
                elif sim.shape[0] == bias_value.shape[0] * h * 2:
                    bias = repeat(bias_value, "b n -> (b h) 1 n", h=h)
                    bias_full = torch.zeros_like(sim[:, :1, :])
                    bias_full[bias.shape[0]:] = bias
                    sim = sim + bias_full
                else:
                    raise RuntimeError(
                        f"Cannot broadcast dynamic bias batch {bias_value.shape[0]} to attention batch-heads {sim.shape[0]}."
                    )

            if exists(mask):
                mask = rearrange(mask, "b ... -> b (...)")
                max_neg_value = -torch.finfo(sim.dtype).max
                mask = repeat(mask, "b j -> (b h) () j", h=h)
                sim.masked_fill_(~mask, max_neg_value)

            attn = sim.softmax(dim=-1)
            out = torch.einsum("b i j, b j d -> b i d", attn, v)
            out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
            return self.to_out(out)

        CrossAttention.forward = dynamic_forward
        try:
            yield
        finally:
            CrossAttention.forward = original_forward

    return _patched()


def patch_p_losses(model, dynamic_state):
    original_p_losses = model.p_losses

    def patched_p_losses(x_start, cond, t, noise=None):
        dynamic_state.start_batch(t, model.num_timesteps)
        return original_p_losses(x_start, cond, t, noise=noise)

    model.p_losses = patched_p_losses
    return original_p_losses


def prepare_config(args, checkpoint_payload):
    from config import Config_Generative_Model

    if "config" in checkpoint_payload:
        config = copy.deepcopy(checkpoint_payload["config"])
    else:
        config = Config_Generative_Model()

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
    if not hasattr(config, "eval_avg"):
        config.eval_avg = True
    if not hasattr(config, "global_pool"):
        config.global_pool = False
    if not hasattr(config, "use_time_cond"):
        config.use_time_cond = True
    if not hasattr(config, "crop_ratio"):
        config.crop_ratio = 0.2
    if not hasattr(config, "img_size"):
        config.img_size = 512
    if not hasattr(config, "ddim_steps"):
        config.ddim_steps = 250
    if not hasattr(config, "HW"):
        config.HW = None

    config.dynamic_chunking = {
        "enabled": True,
        "chunk_count": 3,
        "same_boundaries_as_fixed_chunking": True,
        "router_hidden_dim": args.router_hidden_dim,
        "router_time_dim": args.router_time_dim,
        "router_init_alpha": args.router_init_alpha,
        "router_max_alpha": args.router_max_alpha,
        "entropy_weight": args.entropy_weight,
        "alpha_l2_weight": args.alpha_l2_weight,
        "train_scope": args.train_scope,
    }
    return config


def configure_trainable_params(model, train_scope):
    for param in model.parameters():
        param.requires_grad = False

    for param in model.dynamic_chunk_router.parameters():
        param.requires_grad = True

    if train_scope in {"router_conditioner", "router_crossattn_conditioner"}:
        for param in model.cond_stage_model.parameters():
            param.requires_grad = True

    if train_scope == "router_crossattn_conditioner":
        for name, param in model.named_parameters():
            if "attn2" in name or "time_embed_condtion" in name or "norm2" in name:
                param.requires_grad = True

    trainable = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable parameters selected.")
    return trainable


def build_optimizer(model, args):
    import torch

    trainable = configure_trainable_params(model, args.train_scope)
    router_ids = {id(param) for param in model.dynamic_chunk_router.parameters()}
    router_params = [param for _, param in trainable if id(param) in router_ids]
    other_params = [param for _, param in trainable if id(param) not in router_ids]

    groups = [{"params": router_params, "lr": args.router_lr or args.lr, "name": "dynamic_router"}]
    if other_params:
        groups.append({"params": other_params, "lr": args.lr, "name": "base_tuned_params"})

    print(f"train_scope: {args.train_scope}", flush=True)
    print(f"trainable tensors: {len(trainable)}", flush=True)
    print(f"router trainable tensors: {len(router_params)}", flush=True)
    print(f"other trainable tensors: {len(other_params)}", flush=True)
    print(f"lr: {args.lr}", flush=True)
    print(f"router_lr: {args.router_lr or args.lr}", flush=True)
    return torch.optim.AdamW(groups)


def save_checkpoint(path, model, config, optimizer, dynamic_state, run_step, total_step,
                    start_step, target_step, source_checkpoint, device):
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    config_to_save = copy.deepcopy(config)
    config_to_save.logger = None
    config_to_save.trainer = None
    rng_state = torch.cuda.get_rng_state(device) if device.type == "cuda" else torch.random.get_rng_state()
    payload = {
        "model_state_dict": model.state_dict(),
        "config": config_to_save,
        "state": rng_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "global_step": total_step,
        "total_step": total_step,
        "run_step": run_step,
        "start_step": start_step,
        "target_step": target_step,
        "source_checkpoint": str(source_checkpoint),
        "dynamic_chunking": {
            "chunk_boundaries": dynamic_state.bounds,
            "token_count": dynamic_state.token_count,
            "router_alpha": float(model.dynamic_chunk_router.alpha().detach().cpu()),
            "metadata": config_to_save.dynamic_chunking,
        },
    }
    torch.save(payload, path)
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
    require_file(resolve_path(root, args.eeg_signals_path), "EEG signals")
    require_file(resolve_path(root, args.splits_path), "dataset split")
    require_file(resolve_path(root, args.config_patch), "Stable Diffusion config")
    require_file(resolve_path(root, args.imagenet_path), "ImageNet image directory")

    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    from dataset import create_EEG_dataset
    from dc_ldm.ldm_for_eeg import eLDM_eval

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but torch.cuda.is_available() is False.")

    checkpoint_payload = torch.load(checkpoint_path, map_location="cpu")
    if "model_state_dict" not in checkpoint_payload:
        raise KeyError(f"{checkpoint_path} does not contain key 'model_state_dict'.")

    start_step, target_step, run_steps, inferred_step, inferred_from = resolve_training_window(args, checkpoint_payload)
    config = prepare_config(args, checkpoint_payload)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    print(f"root: {root}", flush=True)
    print(f"device: {device}", flush=True)
    print(f"loading checkpoint: {checkpoint_path}", flush=True)
    print(f"checkpoint step inferred from: {inferred_from or 'none'} ({inferred_step})", flush=True)
    print(f"start_step: {start_step}", flush=True)
    print(f"target_step: {target_step}", flush=True)
    print(f"run_steps_this_invocation: {run_steps}", flush=True)
    print(f"output_dir: {output_dir}", flush=True)
    print(f"subject: {config.subject}", flush=True)
    print(f"batch_size: {config.batch_size}", flush=True)
    print(f"clip_tune: {config.clip_tune}", flush=True)
    print(f"cls_tune: {config.cls_tune}", flush=True)
    print(f"entropy_weight: {args.entropy_weight}", flush=True)
    print(f"alpha_l2_weight: {args.alpha_l2_weight}", flush=True)

    if config.dataset != "EEG":
        raise NotImplementedError("train_dynamic_chunking.py currently supports --dataset EEG only.")

    image_transform = make_train_transform(config)
    dataset_train, _ = create_EEG_dataset(
        eeg_signals_path=config.eeg_signals_path,
        splits_path=config.splits_path,
        imagenet_path=str(resolve_path(root, args.imagenet_path)),
        image_transform=[image_transform, image_transform],
        subject=config.subject,
    )
    print(f"train samples: {len(dataset_train)}", flush=True)
    steps_per_epoch = (len(dataset_train) + config.batch_size - 1) // config.batch_size
    print(f"steps_per_epoch_at_current_batch_size: {steps_per_epoch}", flush=True)
    if steps_per_epoch:
        print(f"start_epoch_equivalent: {start_step / steps_per_epoch:.3f}", flush=True)
        print(f"target_epoch_equivalent: {target_step / steps_per_epoch:.3f}", flush=True)

    print("building eLDM_eval and attaching dynamic chunk router ...", flush=True)
    generative_model = eLDM_eval(
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
    model = generative_model.model
    router = build_dynamic_chunk_router(
        context_dim=768,
        hidden_dim=args.router_hidden_dim,
        time_dim=args.router_time_dim,
        init_alpha=args.router_init_alpha,
        max_alpha=args.router_max_alpha,
    )
    model.dynamic_chunk_router = router
    missing, unexpected = model.load_state_dict(checkpoint_payload["model_state_dict"], strict=False)
    print(f"loaded checkpoint with strict=False; missing={len(missing)}, unexpected={len(unexpected)}", flush=True)
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

    dynamic_state = DynamicRoutingState(model.dynamic_chunk_router, eps=1e-8)
    optimizer = build_optimizer(model, args)
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
    if len(dataloader) == 0:
        raise RuntimeError("training dataloader is empty.")

    original_p_losses = patch_p_losses(model, dynamic_state)
    model.train()
    if model.cond_stage_model is not None:
        model.cond_stage_model.train()

    run_step = 0
    data_iter = iter(dataloader)
    use_total_step_names = args.start_step is not None or args.target_step is not None
    empty_batch_count = 0
    try:
        with patch_cross_attention(dynamic_state):
            while run_step < run_steps:
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(dataloader)
                    batch = next(data_iter)

                if batch is None:
                    empty_batch_count += 1
                    if empty_batch_count > len(dataloader):
                        raise RuntimeError("All batches were empty after skipping bad images.")
                    print("skipped empty batch after filtering bad images", flush=True)
                    continue
                empty_batch_count = 0

                batch = move_to_device(batch, device)
                if run_step == 0:
                    check_batch_devices(batch, device)
                optimizer.zero_grad(set_to_none=True)
                loss, loss_dict = model.shared_step(batch)
                reg_loss, reg_terms = dynamic_state.regularization_loss(args.entropy_weight, args.alpha_l2_weight)
                if reg_terms:
                    loss = loss + reg_loss
                    loss_dict.update({key: value for key, value in reg_terms.items()})
                    loss_dict["dynamic/reg_loss"] = reg_loss
                loss.backward()
                trainable_params = [param for param in model.parameters() if param.requires_grad]
                torch.nn.utils.clip_grad_norm_(trainable_params, 0.5)
                optimizer.step()

                run_step += 1
                total_step = start_step + run_step
                parts = [
                    f"run_step {run_step}/{run_steps}",
                    f"total_step {total_step}/{target_step}",
                    f"loss={tensor_to_float(loss):.6f}",
                ]
                for key, value in sorted(loss_dict.items()):
                    parts.append(f"{key}={tensor_to_float(value):.6f}")
                for key, value in sorted(dynamic_state.stats().items()):
                    parts.append(f"{key}={float(value):.6f}")
                print(" | ".join(parts), flush=True)

                if args.save_every_n_steps and run_step % args.save_every_n_steps == 0:
                    name_step = total_step if use_total_step_names else run_step
                    filename = (
                        f"dynamic_total_{name_step}steps.pth"
                        if use_total_step_names
                        else f"dynamic_{name_step}steps.pth"
                    )
                    save_checkpoint(
                        output_dir / filename,
                        model,
                        config,
                        optimizer,
                        dynamic_state,
                        run_step,
                        total_step,
                        start_step,
                        target_step,
                        checkpoint_path,
                        device,
                    )
    finally:
        model.p_losses = original_p_losses

    final_name = (
        f"dynamic_total_{target_step}steps.pth"
        if use_total_step_names
        else f"dynamic_{run_steps}steps.pth"
    )
    final_path = output_dir / final_name
    save_checkpoint(final_path, model, config, optimizer, dynamic_state, run_step, target_step,
                    start_step, target_step, checkpoint_path, device)
    print(f"final dynamic checkpoint: {final_path}", flush=True)
    print(
        "Note: original gen_eval_eeg.py will ignore dynamic_chunk_router keys. "
        "Use a dynamic inference script/patch to evaluate learned routing.",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", flush=True)
        raise SystemExit(1)
