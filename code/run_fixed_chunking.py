import argparse
import contextlib
import datetime
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image


CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))


def parse_limit(value):
    if value is None or value == "all":
        return None
    value = int(value)
    if value <= 0:
        return None
    return value


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fixed EEG chunk routing experiments for DreamDiffusion inference."
    )
    parser.add_argument("--stage", choices=["sample", "train"], default="sample")
    parser.add_argument(
        "--routing-mode",
        choices=["baseline", "uniform", "forward", "reverse", "constant1", "constant2", "constant3"],
        default="baseline",
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--dataset", type=str, default="EEG")
    parser.add_argument("--model_path", "--checkpoint", dest="model_path", type=Path, required=False,
                        default=Path("pretrains/eeg_pretrain/checkpoint.pth"))
    parser.add_argument("--splits_path", type=Path, default=Path("datasets/block_splits_by_image_single.pth"))
    parser.add_argument("--eeg_signals_path", type=Path, default=Path("datasets/eeg_5_95_std.pth"))
    parser.add_argument("--config_patch", type=Path, default=Path("pretrains/models/config15.yaml"))
    parser.add_argument("--imagenet_path", type=Path, default=Path("datasets/imageNet_images"))
    parser.add_argument("--subject", type=int, default=4)
    parser.add_argument("--limit", default=None, help="1, 5, all, or omit for all.")
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument("--output-dir", type=Path, default=Path("results/fixed_chunking"))
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--num-sampling-steps", type=int, default=None)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--debug-shapes", action="store_true")
    parser.add_argument("--save-routing-metadata", action="store_true")
    parser.add_argument("--run-tests", action="store_true", help="Run routing math tests and exit.")
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument(
        "--time-token-policy",
        choices=["auto", "post_projector", "preserve"],
        default="auto",
        help=(
            "auto/post_projector applies routing to the original 77-token U-Net conditioning sequence. "
            "preserve is kept only for documentation and is not supported by the official checkpoint."
        ),
    )
    return parser.parse_args()


def resolve_path(root, path):
    path = Path(path)
    return path if path.is_absolute() else root / path


def require_path(path, label):
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def chunk_boundaries(num_tokens, chunks=3):
    if num_tokens < chunks:
        raise ValueError(f"Need at least {chunks} tokens, got {num_tokens}.")
    base = num_tokens // chunks
    rem = num_tokens % chunks
    lengths = [base + (1 if i < rem else 0) for i in range(chunks)]
    bounds = []
    start = 0
    for length in lengths:
        end = start + length
        bounds.append((start, end))
        start = end
    return bounds


def membership_matrix(num_tokens, bounds, device=None, dtype=None):
    import torch

    matrix = torch.zeros((len(bounds), num_tokens), device=device, dtype=dtype or torch.float32)
    for idx, (start, end) in enumerate(bounds):
        matrix[idx, start:end] = 1.0
    return matrix


def validate_chunks(num_tokens, bounds):
    covered = []
    for start, end in bounds:
        if not (0 <= start < end <= num_tokens):
            raise AssertionError(f"Invalid chunk boundary: {(start, end)} for N={num_tokens}.")
        covered.extend(range(start, end))
    if sorted(covered) != list(range(num_tokens)):
        raise AssertionError(f"Chunks do not exactly cover all tokens for N={num_tokens}: {bounds}")
    if len(set(covered)) != len(covered):
        raise AssertionError(f"Chunks overlap for N={num_tokens}: {bounds}")
    lengths = [end - start for start, end in bounds]
    if min(lengths) <= 0:
        raise AssertionError(f"Empty chunk detected: {bounds}")
    if max(lengths) - min(lengths) > 1:
        raise AssertionError(f"Chunk lengths differ by more than 1: {lengths}")


def get_chunk_weights(routing_mode, progress, device=None, dtype=None):
    import torch

    if routing_mode == "baseline":
        return None

    if torch.is_tensor(progress):
        progress_tensor = progress.to(device=device or progress.device, dtype=torch.float32)
        scalar_input = progress_tensor.ndim == 0
    else:
        progress_tensor = torch.tensor(progress, device=device, dtype=torch.float32)
        scalar_input = True

    progress_flat = progress_tensor.reshape(-1)
    weights = torch.empty((progress_flat.shape[0], 3), device=progress_flat.device, dtype=torch.float32)

    if routing_mode == "uniform":
        weights[:] = torch.tensor([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], device=weights.device)
    elif routing_mode == "forward":
        early = progress_flat < (1.0 / 3.0)
        middle = (progress_flat >= (1.0 / 3.0)) & (progress_flat < (2.0 / 3.0))
        late = progress_flat >= (2.0 / 3.0)
        weights[early] = torch.tensor([0.8, 0.1, 0.1], device=weights.device)
        weights[middle] = torch.tensor([0.1, 0.8, 0.1], device=weights.device)
        weights[late] = torch.tensor([0.1, 0.1, 0.8], device=weights.device)
    elif routing_mode == "reverse":
        early = progress_flat < (1.0 / 3.0)
        middle = (progress_flat >= (1.0 / 3.0)) & (progress_flat < (2.0 / 3.0))
        late = progress_flat >= (2.0 / 3.0)
        weights[early] = torch.tensor([0.1, 0.1, 0.8], device=weights.device)
        weights[middle] = torch.tensor([0.1, 0.8, 0.1], device=weights.device)
        weights[late] = torch.tensor([0.8, 0.1, 0.1], device=weights.device)
    elif routing_mode == "constant1":
        weights[:] = torch.tensor([0.8, 0.1, 0.1], device=weights.device)
    elif routing_mode == "constant2":
        weights[:] = torch.tensor([0.1, 0.8, 0.1], device=weights.device)
    elif routing_mode == "constant3":
        weights[:] = torch.tensor([0.1, 0.1, 0.8], device=weights.device)
    else:
        raise ValueError(routing_mode)

    if not torch.allclose(weights.sum(dim=-1), torch.ones(weights.shape[0], device=weights.device), atol=1e-6):
        raise AssertionError(f"{routing_mode} weights do not sum to 1: {weights}")
    if torch.any(weights <= 0):
        raise AssertionError(f"{routing_mode} weights contain non-positive values: {weights}")
    if torch.isnan(weights).any() or torch.isinf(weights).any():
        raise AssertionError(f"{routing_mode} weights contain NaN or Inf: {weights}")

    weights = weights.to(dtype=dtype or torch.float32)
    return weights[0] if scalar_input else weights.reshape(*progress_tensor.shape, 3)


def stage_weights(mode, progress):
    if mode == "baseline":
        return None
    return get_chunk_weights(mode, progress).detach().cpu().tolist()


def validate_stage_weights():
    checks = {
        "uniform@0": stage_weights("uniform", 0.0),
        "forward@early": stage_weights("forward", 0.0),
        "forward@middle": stage_weights("forward", 0.5),
        "forward@late": stage_weights("forward", 1.0),
        "forward@one_third": stage_weights("forward", 1.0 / 3.0),
        "forward@two_thirds": stage_weights("forward", 2.0 / 3.0),
        "reverse@early": stage_weights("reverse", 0.1),
        "reverse@middle": stage_weights("reverse", 0.5),
        "reverse@late": stage_weights("reverse", 0.9),
        "constant1@early": stage_weights("constant1", 0.1),
        "constant1@middle": stage_weights("constant1", 0.5),
        "constant1@late": stage_weights("constant1", 0.9),
        "constant2@early": stage_weights("constant2", 0.1),
        "constant2@middle": stage_weights("constant2", 0.5),
        "constant2@late": stage_weights("constant2", 0.9),
        "constant3@early": stage_weights("constant3", 0.1),
        "constant3@middle": stage_weights("constant3", 0.5),
        "constant3@late": stage_weights("constant3", 0.9),
    }
    expected = {
        "uniform@0": [1.0 / 3.0] * 3,
        "forward@early": [0.8, 0.1, 0.1],
        "forward@middle": [0.1, 0.8, 0.1],
        "forward@late": [0.1, 0.1, 0.8],
        "forward@one_third": [0.1, 0.8, 0.1],
        "forward@two_thirds": [0.1, 0.1, 0.8],
        "reverse@early": [0.1, 0.1, 0.8],
        "reverse@middle": [0.1, 0.8, 0.1],
        "reverse@late": [0.8, 0.1, 0.1],
        "constant1@early": [0.8, 0.1, 0.1],
        "constant1@middle": [0.8, 0.1, 0.1],
        "constant1@late": [0.8, 0.1, 0.1],
        "constant2@early": [0.1, 0.8, 0.1],
        "constant2@middle": [0.1, 0.8, 0.1],
        "constant2@late": [0.1, 0.8, 0.1],
        "constant3@early": [0.1, 0.1, 0.8],
        "constant3@middle": [0.1, 0.1, 0.8],
        "constant3@late": [0.1, 0.1, 0.8],
    }
    for key, weights in checks.items():
        if not np.allclose(weights, expected[key]):
            raise AssertionError(f"{key} got {weights}, expected {expected[key]}")
        if not math.isclose(sum(weights), 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise AssertionError(f"{key} weights do not sum to 1: {weights}")
    return checks


def validate_token_prior():
    import torch

    bounds = chunk_boundaries(77, 3)
    validate_chunks(77, bounds)
    membership = membership_matrix(77, bounds, device="cpu", dtype=torch.float32)
    checks = {}
    for mode in ["uniform", "forward", "reverse", "constant1", "constant2", "constant3"]:
        for progress in [0.0, 0.1, 0.5, 0.9, 1.0]:
            weights = get_chunk_weights(mode, progress, device=torch.device("cpu"), dtype=torch.float32)
            pi = weights @ membership
            bias = torch.log(pi + 1e-8)
            if torch.any(pi <= 0):
                raise AssertionError(f"{mode}@{progress} token prior has non-positive values.")
            if torch.isnan(bias).any() or torch.isinf(bias).any():
                raise AssertionError(f"{mode}@{progress} log-bias contains NaN or Inf.")
            checks[f"{mode}@{progress}"] = {
                "pi_min": float(pi.min().item()),
                "pi_max": float(pi.max().item()),
            }

    batch_progress = torch.tensor([0.1, 0.5, 0.9])
    batch_weights = get_chunk_weights("reverse", batch_progress)
    if tuple(batch_weights.shape) != (3, 3):
        raise AssertionError(f"batched progress returned wrong shape: {tuple(batch_weights.shape)}")
    if not torch.allclose(batch_weights.sum(dim=-1), torch.ones(3), atol=1e-6):
        raise AssertionError("batched reverse weights do not sum to 1.")
    return checks


def uniform_equivalence_test():
    import torch

    torch.manual_seed(0)
    q = torch.randn(4, 13, 32, dtype=torch.float32)
    k = torch.randn(4, 17, 32, dtype=torch.float32)
    v = torch.randn(4, 17, 32, dtype=torch.float32)
    sim = torch.einsum("b i d, b j d -> b i j", q, k) / math.sqrt(q.shape[-1])
    baseline = sim.softmax(dim=-1).matmul(v)
    uniform = (sim + math.log(1.0 / 3.0)).softmax(dim=-1).matmul(v)
    return float((baseline - uniform).abs().max().item())


class RoutingState:
    def __init__(self, mode, eps):
        self.mode = mode
        self.eps = eps
        self.enabled = mode != "baseline"
        self.progress = 0.0
        self.total_steps = None
        self.bounds = None
        self.token_count = None
        self.membership = None
        self.bias = None
        self.conditional_batch = None
        self.debug_shapes = {}
        self.printed_attention_debug = False
        self.first_progress = None
        self.last_progress = None
        self.cfg_mode = "no CFG detected in this script unless sampler receives unconditional_conditioning"

    def set_token_count(self, token_count, device, dtype):
        import torch

        self.token_count = int(token_count)
        self.bounds = chunk_boundaries(self.token_count, 3)
        validate_chunks(self.token_count, self.bounds)
        self.membership = membership_matrix(self.token_count, self.bounds, device=device, dtype=torch.float32)
        self.update_bias(device=device, dtype=dtype)

    def update_progress(self, step_index, total_steps, device=None, dtype=None):
        self.total_steps = int(total_steps)
        self.progress = 0.0 if total_steps <= 1 else float(step_index) / float(total_steps - 1)
        if self.first_progress is None:
            self.first_progress = self.progress
        self.last_progress = self.progress
        if self.membership is not None:
            self.update_bias(device=device or self.membership.device, dtype=dtype or self.membership.dtype)

    def update_bias(self, device, dtype):
        import torch

        if not self.enabled or self.membership is None:
            self.bias = None
            return
        weights = get_chunk_weights(self.mode, self.progress, device=device, dtype=torch.float32)
        if not torch.allclose(weights.sum(dim=-1), torch.ones_like(weights.sum(dim=-1)), atol=1e-6):
            raise AssertionError(f"stage weights do not sum to 1: {weights.detach().cpu().tolist()}")
        pi = weights @ self.membership.to(device=device)
        if torch.any(pi <= 0):
            raise AssertionError(f"pi contains non-positive values: min={pi.min().item()}")
        bias = torch.log(pi + self.eps)
        if torch.isnan(bias).any() or torch.isinf(bias).any():
            raise AssertionError("log(pi + eps) contains NaN or Inf.")
        self.bias = bias.to(dtype=dtype)

    def token_prior_stats(self):
        if self.bias is None:
            return None
        pi = self.bias.float().exp()
        return {
            "pi_min": float(pi.min().item()),
            "pi_max": float(pi.max().item()),
            "bias_shape": list(self.bias.shape),
            "progress": self.progress,
            "weights": stage_weights(self.mode, self.progress),
        }

    def progress_stats(self):
        return {
            "first_sampler_progress": self.first_progress,
            "last_sampler_progress": self.last_progress,
        }


def patch_cond_stage_for_time_tokens(cond_stage, routing_state):
    original_forward = cond_stage.forward

    def forward_time_preserving(x):
        latent_time = cond_stage.mae(x)
        routing_state.debug_shapes.setdefault("raw_eeg_shape", list(x.shape))
        routing_state.debug_shapes.setdefault("eeg_encoder_output_shape", list(latent_time.shape))
        routing_state.debug_shapes.setdefault("channel_mapper_original", hasattr(cond_stage, "channel_mapper"))
        if routing_state.token_count is None:
            routing_state.set_token_count(latent_time.shape[1], latent_time.device, latent_time.dtype)
        projected = cond_stage.dim_mapper(latent_time)
        routing_state.debug_shapes.setdefault("eeg_projector_output_shape", list(projected.shape))
        routing_state.debug_shapes.setdefault("cross_attention_context_shape", list(projected.shape))
        return projected, latent_time

    cond_stage.forward = forward_time_preserving
    return original_forward


def record_original_conditioner_shapes(cond_stage, conditioning, re_latent, routing_state):
    routing_state.debug_shapes.setdefault("eeg_encoder_return_shape", list(re_latent.shape))
    routing_state.debug_shapes.setdefault("cross_attention_context_shape", list(conditioning.shape))
    routing_state.debug_shapes.setdefault("conditioner_global_pool", bool(getattr(cond_stage, "global_pool", False)))
    routing_state.debug_shapes.setdefault("conditioner_has_channel_mapper", hasattr(cond_stage, "channel_mapper"))


def routing_weight_summary(mode):
    if mode == "baseline":
        return None
    return {
        "early": stage_weights(mode, 0.1),
        "middle": stage_weights(mode, 0.5),
        "late": stage_weights(mode, 0.9),
    }


def constant_chunk(mode):
    if mode.startswith("constant"):
        return int(mode.replace("constant", ""))
    return None


@contextlib.contextmanager
def patch_cross_attention(routing_state, debug_shapes=False):
    import torch
    from einops import rearrange, repeat
    from dc_ldm.modules.attention import CrossAttention, default, exists

    original_forward = CrossAttention.forward

    def routed_forward(self, x, context=None, mask=None):
        h = self.heads
        q = self.to_q(x)
        context = default(context, x)
        is_cross = context is not x and context is not None
        k = self.to_k(context)
        v = self.to_v(context)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v))
        sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale

        if routing_state.enabled and is_cross:
            if routing_state.bias is None:
                raise RuntimeError("Routing bias is not initialized.")
            if sim.shape[-1] != routing_state.bias.shape[-1]:
                raise RuntimeError(
                    f"Cross-attention context length {sim.shape[-1]} does not match routing token count "
                    f"{routing_state.bias.shape[-1]}. This usually means the original projector changed token length."
                )
            batch_heads = sim.shape[0]
            bias_value = routing_state.bias.to(device=sim.device, dtype=sim.dtype)
            if bias_value.ndim == 1:
                bias = bias_value.view(1, 1, -1)
                conditional_bias = bias
            elif bias_value.ndim == 2:
                if routing_state.conditional_batch is None:
                    raise RuntimeError("Batch routing bias requires routing_state.conditional_batch.")
                if bias_value.shape[0] != routing_state.conditional_batch:
                    raise RuntimeError(
                        f"Batch routing bias has batch {bias_value.shape[0]}, "
                        f"but conditional batch is {routing_state.conditional_batch}."
                    )
                bias = repeat(bias_value, "b n -> (b h) 1 n", h=h)
                conditional_bias = bias
            else:
                raise RuntimeError(f"Routing bias must be 1D or 2D, got shape {list(bias_value.shape)}.")

            if routing_state.conditional_batch is not None and batch_heads == routing_state.conditional_batch * h * 2:
                bias_full = torch.zeros_like(sim[:, :1, :])
                bias_full[routing_state.conditional_batch * h:] = conditional_bias
                sim = sim + bias_full
                routing_state.cfg_mode = "detected [unconditional; conditional] concatenation, applied bias only to conditional branch"
            else:
                sim = sim + bias
                routing_state.cfg_mode = "no unconditional branch in current call, applied bias to all EEG-conditioned batch items"

            if debug_shapes and not routing_state.printed_attention_debug:
                routing_state.debug_shapes["attention_q_shape"] = list(q.shape)
                routing_state.debug_shapes["attention_k_shape"] = list(k.shape)
                routing_state.debug_shapes["attention_v_shape"] = list(v.shape)
                routing_state.debug_shapes["attention_logits_shape"] = list(sim.shape)
                routing_state.debug_shapes["attention_bias_shape"] = list(bias.shape)
                routing_state.printed_attention_debug = True

        if exists(mask):
            mask = rearrange(mask, "b ... -> b (...)")
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, "b j -> (b h) () j", h=h)
            sim.masked_fill_(~mask, max_neg_value)

        attn = sim.softmax(dim=-1)
        out = torch.einsum("b i j, b j d -> b i d", attn, v)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
        return self.to_out(out)

    CrossAttention.forward = routed_forward
    try:
        yield
    finally:
        CrossAttention.forward = original_forward


class RoutingPLMSSampler:
    def __init__(self, model, routing_state):
        from dc_ldm.models.diffusion.plms import PLMSSampler

        self.inner = PLMSSampler(model)
        self.routing_state = routing_state

    def sample(self, S, *args, **kwargs):
        original_p_sample = self.inner.p_sample_plms

        def wrapped_p_sample(x, c, t, index, *p_args, **p_kwargs):
            total = int(S)
            step_index = total - int(index) - 1
            self.routing_state.update_progress(step_index, total, device=x.device, dtype=x.dtype)
            return original_p_sample(x, c, t, index, *p_args, **p_kwargs)

        self.inner.p_sample_plms = wrapped_p_sample
        try:
            result = self.inner.sample(S, *args, **kwargs)
            if S > 1:
                self.routing_state.update_progress(S - 1, S)
            return result
        finally:
            self.inner.p_sample_plms = original_p_sample


def build_model_and_data(args, routing_state):
    root = args.root.resolve()
    model_path = resolve_path(root, args.model_path)
    splits_path = resolve_path(root, args.splits_path)
    eeg_path = resolve_path(root, args.eeg_signals_path)
    config_patch = resolve_path(root, args.config_patch)
    imagenet_path = resolve_path(root, args.imagenet_path)
    for path, label in [
        (model_path, "generation checkpoint"),
        (splits_path, "split file"),
        (eeg_path, "EEG signals"),
        (config_patch, "Stable Diffusion config"),
        (imagenet_path, "ImageNet image directory"),
    ]:
        require_path(path, label)

    import torch
    import torchvision.transforms as transforms
    from einops import rearrange

    from dataset import create_EEG_dataset
    from dc_ldm.ldm_for_eeg import eLDM_eval

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    sd = torch.load(model_path, map_location="cpu")
    config = sd["config"]
    config.root_path = str(root)
    if args.num_sampling_steps is not None:
        config.ddim_steps = args.num_sampling_steps
    if args.num_samples is not None:
        config.num_samples = args.num_samples

    def normalize(img):
        if img.shape[-1] == 3:
            img = rearrange(img, "h w c -> c h w")
        img = torch.tensor(img)
        return img * 2.0 - 1.0

    def channel_last(img):
        if img.shape[-1] == 3:
            return img
        return rearrange(img, "c h w -> h w c")

    transform = transforms.Compose([normalize, transforms.Resize((512, 512)), channel_last])
    _, dataset_test = create_EEG_dataset(
        eeg_signals_path=str(eeg_path),
        splits_path=str(splits_path),
        imagenet_path=str(imagenet_path),
        image_transform=[transform, transform],
        subject=args.subject,
    )
    num_voxels = dataset_test.dataset.data_len
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable.")

    model = eLDM_eval(
        str(config_patch),
        num_voxels,
        device=device,
        pretrain_root=config.pretrain_gm_path,
        logger=None,
        ddim_steps=config.ddim_steps,
        global_pool=config.global_pool,
        use_time_cond=config.use_time_cond,
    )
    missing, unexpected = model.model.load_state_dict(sd["model_state_dict"], strict=False)
    print(f"loaded checkpoint: {model_path}", flush=True)
    print(f"load_state_dict strict=False missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    state = sd.get("state")
    return model, dataset_test, config, state, {
        "checkpoint_path": str(model_path),
        "splits_path": str(splits_path),
        "eeg_signals_path": str(eeg_path),
        "config_path": str(config_patch),
        "imagenet_path": str(imagenet_path),
    }


def save_image_grid_and_samples(generative_model, dataset, config, args, routing_state, output_dir, state):
    import torch
    from einops import rearrange, repeat
    from torchvision.utils import make_grid

    from dc_ldm.models.diffusion.plms import PLMSSampler

    limit = parse_limit(args.limit)
    model = generative_model.model.to(generative_model.device)
    model.eval()
    sampler = PLMSSampler(model) if args.routing_mode == "baseline" else RoutingPLMSSampler(model, routing_state)

    if config.HW is None:
        shape = (generative_model.ldm_config.model.params.channels, generative_model.ldm_config.model.params.image_size, generative_model.ldm_config.model.params.image_size)
    else:
        num_resolutions = len(generative_model.ldm_config.model.params.first_stage_config.params.ddconfig.ch_mult)
        shape = (
            generative_model.ldm_config.model.params.channels,
            config.HW[0] // 2 ** (num_resolutions - 1),
            config.HW[1] // 2 ** (num_resolutions - 1),
        )

    if state is not None and generative_model.device.type == "cuda":
        try:
            torch.cuda.set_rng_state(state)
        except RuntimeError as exc:
            print(f"[WARN] skipping CUDA RNG state restore: {exc}", flush=True)

    all_samples = []
    attention_context = (
        contextlib.nullcontext()
        if args.routing_mode == "baseline"
        else patch_cross_attention(routing_state, debug_shapes=args.debug_shapes)
    )
    with torch.no_grad(), model.ema_scope(), attention_context:
        for count, item in enumerate(dataset):
            if limit is not None and count >= limit:
                break
            latent = item["eeg"]
            gt_image = rearrange(item["image"], "h w c -> 1 c h w")
            print(f"rendering sample {count}, {config.num_samples} examples in {config.ddim_steps} steps.", flush=True)
            c, re_latent = model.get_learned_conditioning(
                repeat(latent, "h w -> c h w", c=config.num_samples).to(generative_model.device)
            )
            if routing_state.enabled:
                record_original_conditioner_shapes(model.cond_stage_model, c, re_latent, routing_state)
                if routing_state.token_count is None:
                    routing_state.set_token_count(c.shape[1], c.device, c.dtype)
            routing_state.conditional_batch = c.shape[0]
            if args.debug_shapes:
                routing_state.debug_shapes.setdefault("actual_unet_context_shape", list(c.shape))
                routing_state.debug_shapes.setdefault("sampler_name", "PLMS")
            samples_ddim, _ = sampler.sample(
                S=config.ddim_steps,
                conditioning=c,
                batch_size=config.num_samples,
                shape=shape,
                verbose=False,
            )
            x_samples = model.decode_first_stage(samples_ddim)
            x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
            gt_image = torch.clamp((gt_image + 1.0) / 2.0, min=0.0, max=1.0)
            packed = torch.cat([gt_image, x_samples.detach().cpu()], dim=0)
            all_samples.append(packed)
            samples_t = (255.0 * packed.numpy()).astype(np.uint8)
            for copy_idx, img_t in enumerate(samples_t):
                img_t = rearrange(img_t, "c h w -> h w c")
                Image.fromarray(img_t).save(output_dir / f"test{count}-{copy_idx}.png")

    if not all_samples:
        raise RuntimeError("No samples generated. Check --limit and dataset.")

    grid = torch.stack(all_samples, 0)
    grid = rearrange(grid, "n b c h w -> (n b) c h w")
    grid = make_grid(grid, nrow=config.num_samples + 1)
    grid = (255.0 * rearrange(grid, "c h w -> h w c").cpu().numpy()).astype(np.uint8)
    Image.fromarray(grid).save(output_dir / "samples_test.png")
    model.to("cpu")
    return len(all_samples)


def git_commit(root):
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except Exception:
        return None


def write_metadata(path, args, routing_state, config, generated_count, paths):
    metadata = {
        "routing_mode": args.routing_mode,
        "stage": args.stage,
        "time_token_policy": args.time_token_policy,
        "chunk_boundaries": routing_state.bounds,
        "chunk_lengths": [end - start for start, end in routing_state.bounds] if routing_state.bounds else None,
        "stage_weights": routing_weight_summary(args.routing_mode),
        "all_mode_stage_weights": {
            "uniform": routing_weight_summary("uniform"),
            "forward": routing_weight_summary("forward"),
            "reverse": routing_weight_summary("reverse"),
            "constant1": routing_weight_summary("constant1"),
            "constant2": routing_weight_summary("constant2"),
            "constant3": routing_weight_summary("constant3"),
        },
        "constant_chunk": constant_chunk(args.routing_mode),
        "seed": args.seed,
        "checkpoint_path": paths["checkpoint_path"],
        "checkpoint": paths["checkpoint_path"],
        "config_path": paths["config_path"],
        "config": paths["config_path"],
        "sampling_steps": config.ddim_steps,
        "sampler_name": "PLMS",
        "sampler": "PLMS",
        "dataset_split": paths["splits_path"],
        "number_of_generated_samples": generated_count,
        "number_of_samples": generated_count,
        "num_samples_per_eeg": config.num_samples,
        "git_commit_hash": git_commit(args.root.resolve()),
        "debug_shapes": routing_state.debug_shapes,
        "routing_stats": routing_state.token_prior_stats(),
        "progress_stats": routing_state.progress_stats(),
        "cfg_handling": routing_state.cfg_mode,
        "notes": (
            "The official DreamDiffusion checkpoint maps EEG encoder outputs through channel_mapper/dim_mapper before "
            "U-Net cross-attention. To stay checkpoint-compatible, non-baseline routing modes leave the conditioner "
            "unchanged and apply routing bias to the post-projector U-Net context tokens. These chunks are not raw EEG "
            "time chunks."
        ),
    }
    path.write_text(json.dumps(metadata, indent=2))


def run_tests():
    err = uniform_equivalence_test()
    weights = validate_stage_weights()
    prior_checks = validate_token_prior()
    bounds = chunk_boundaries(110, 3)
    validate_chunks(110, bounds)
    if bounds != [(0, 37), (37, 74), (74, 110)]:
        raise AssertionError(f"N=110 boundaries wrong: {bounds}")
    print(f"uniform equivalence max_abs_error: {err:.8g}", flush=True)
    if err >= 1e-5:
        raise AssertionError(f"uniform equivalence error too high: {err}")
    print(f"forward weights test: {weights}", flush=True)
    print(f"token prior tests: {prior_checks}", flush=True)
    print(f"N=110 chunk boundaries: {bounds}", flush=True)


def main():
    args = parse_args()
    if args.run_tests:
        run_tests()
        return
    if args.stage == "train":
        raise NotImplementedError(
            "Fixed chunking training is intentionally not implemented yet. "
            "This script currently supports inference-time sanity checks only. "
            "Training would require freezing policy changes and validating gradients through the routed cross-attention path."
        )

    root = args.root.resolve()
    output_root = resolve_path(root, args.output_dir)
    timestamp = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    output_dir = output_root / args.routing_mode / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    routing_state = RoutingState(args.routing_mode, args.eps)
    print(f"routing mode: {args.routing_mode}", flush=True)
    if args.routing_mode != "baseline":
        print(f"early weights: {stage_weights(args.routing_mode, 0.1)}", flush=True)
        print(f"middle weights: {stage_weights(args.routing_mode, 0.5)}", flush=True)
        print(f"late weights: {stage_weights(args.routing_mode, 0.9)}", flush=True)
    model, dataset_test, config, state, paths = build_model_and_data(args, routing_state)

    original_cond_forward = None
    if args.routing_mode != "baseline":
        if args.time_token_policy == "preserve":
            raise RuntimeError(
                "--time-token-policy preserve is incompatible with the official checkpoint: "
                "the U-Net was trained to receive the original post-projector conditioning tokens, "
                "not raw EEG encoder tokens."
            )
        print(
            "routing mode: original conditioner is unchanged; routing is applied to post-projector "
            "U-Net context tokens.",
            flush=True,
        )
    else:
        print("baseline mode: original attention and original conditioner are unchanged.", flush=True)

    try:
        generated_count = save_image_grid_and_samples(model, dataset_test, config, args, routing_state, output_dir, state)
    finally:
        if original_cond_forward is not None:
            model.model.cond_stage_model.forward = original_cond_forward

    if args.save_routing_metadata or True:
        write_metadata(output_dir / "metadata.json", args, routing_state, config, generated_count, paths)

    print(f"output_dir: {output_dir}", flush=True)
    if routing_state.bounds:
        print(f"EEG token count N: {routing_state.token_count}", flush=True)
        print(f"chunk boundaries: {routing_state.bounds}", flush=True)
        print(f"chunk lengths: {[e - s for s, e in routing_state.bounds]}", flush=True)
        print(f"first sampler progress: {routing_state.first_progress}", flush=True)
        print(f"last sampler progress: {routing_state.last_progress}", flush=True)
        print(f"routing stats: {routing_state.token_prior_stats()}", flush=True)
    print(f"CFG handling: {routing_state.cfg_mode}", flush=True)
    print(f"metadata: {output_dir / 'metadata.json'}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", flush=True)
        raise SystemExit(1)
