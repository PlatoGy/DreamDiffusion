import argparse
import contextlib
import datetime
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from run_fixed_chunking import (  # noqa: E402
    RoutingState,
    build_model_and_data,
    patch_cross_attention,
    record_original_conditioner_shapes,
    resolve_path,
    routing_weight_summary,
    stage_weights,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export four real denoising progress frames for fixed-chunk forward routing."
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--ckpt", "--model_path", dest="model_path", type=Path,
                        default=Path("pretrains/eeg_pretrain/checkpoint.pth"))
    parser.add_argument("--config", "--config_patch", dest="config_patch", type=Path,
                        default=Path("pretrains/models/config15.yaml"))
    parser.add_argument("--splits_path", type=Path, default=Path("datasets/block_splits_by_image_single.pth"))
    parser.add_argument("--eeg_signals_path", type=Path, default=Path("datasets/eeg_5_95_std.pth"))
    parser.add_argument("--imagenet_path", type=Path, default=Path("datasets/imageNet_images"))
    parser.add_argument("--subject", type=int, default=4)
    parser.add_argument("--routing-mode", choices=["forward"], default="forward")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--prediction-index", type=int, default=0,
                        help="Generated candidate index inside the sampling batch. Default: 0.")
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--num-samples", type=int, default=1,
                        help="Sampling batch size. Use the same value when comparing with run_fixed_chunking.py.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/fixed_chunking_progress"))
    parser.add_argument("--save-progress-grid", dest="save_progress_grid", action="store_true")
    parser.add_argument("--no-save-progress-grid", dest="save_progress_grid", action="store_false")
    parser.add_argument("--save-individual-frames", dest="save_individual_frames", action="store_true")
    parser.add_argument("--no-save-individual-frames", dest="save_individual_frames", action="store_false")
    parser.add_argument("--match-forward-rng-order", dest="match_forward_rng_order", action="store_true",
                        help="Advance RNG so sample-index N uses the same initial noise as run_fixed_chunking.py.")
    parser.add_argument("--no-match-forward-rng-order", dest="match_forward_rng_order", action="store_false")
    parser.set_defaults(
        save_progress_grid=True,
        save_individual_frames=True,
        match_forward_rng_order=True,
    )
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--debug-shapes", action="store_true")
    return parser.parse_args()


def capture_steps(total_steps):
    if total_steps <= 0:
        raise ValueError(f"total_steps must be positive, got {total_steps}.")
    first_stage = max(1, total_steps // 3)
    second_stage = max(first_stage + 1, (2 * total_steps) // 3)
    second_stage = min(second_stage, total_steps)
    return {
        "frame0_noise": 0,
        "frame1_after_chunk1": first_stage,
        "frame2_after_chunk2": second_stage,
        "frame3_final_after_chunk3": total_steps,
    }


class ProgressRoutingPLMSSampler:
    def __init__(self, model, routing_state, captures):
        from dc_ldm.models.diffusion.plms import PLMSSampler

        self.inner = PLMSSampler(model)
        self.routing_state = routing_state
        self.captures = captures
        self.latents = {}

    def sample(self, S, x_T, *args, **kwargs):
        original_p_sample = self.inner.p_sample_plms
        wanted = set(self.captures.values())
        self.latents[0] = x_T.detach().cpu()

        def wrapped_p_sample(x, c, t, index, *p_args, **p_kwargs):
            total = int(S)
            step_index = total - int(index) - 1
            self.routing_state.update_progress(step_index, total, device=x.device, dtype=x.dtype)
            outs = original_p_sample(x, c, t, index, *p_args, **p_kwargs)
            x_prev = outs[0]
            completed_steps = step_index + 1
            if completed_steps in wanted:
                self.latents[completed_steps] = x_prev.detach().cpu()
            return outs

        self.inner.p_sample_plms = wrapped_p_sample
        try:
            result = self.inner.sample(S, x_T=x_T, *args, **kwargs)
            self.routing_state.update_progress(S - 1, S, device=x_T.device, dtype=x_T.dtype)
            self.latents[S] = result[0].detach().cpu()
            return result
        finally:
            self.inner.p_sample_plms = original_p_sample


def latent_shape(generative_model, config):
    if config.HW is None:
        return (
            generative_model.ldm_config.model.params.channels,
            generative_model.ldm_config.model.params.image_size,
            generative_model.ldm_config.model.params.image_size,
        )

    num_resolutions = len(generative_model.ldm_config.model.params.first_stage_config.params.ddconfig.ch_mult)
    return (
        generative_model.ldm_config.model.params.channels,
        config.HW[0] // 2 ** (num_resolutions - 1),
        config.HW[1] // 2 ** (num_resolutions - 1),
    )


def restore_rng_state(state, device):
    import torch

    if state is None:
        return
    try:
        if device.type == "cuda":
            torch.cuda.set_rng_state(state)
        else:
            torch.random.set_rng_state(state)
    except RuntimeError as exc:
        print(f"[WARN] skipping checkpoint RNG state restore: {exc}", flush=True)


def to_uint8_image(tensor_chw):
    from einops import rearrange

    image = torch_clamp_image(tensor_chw)
    image = (255.0 * rearrange(image, "c h w -> h w c").cpu().numpy()).astype(np.uint8)
    return Image.fromarray(image)


def torch_clamp_image(tensor_chw):
    import torch

    return torch.clamp((tensor_chw + 1.0) / 2.0, min=0.0, max=1.0)


def decode_frame(model, latent_batch_cpu, prediction_index, device):
    if prediction_index < 0 or prediction_index >= latent_batch_cpu.shape[0]:
        raise IndexError(
            f"prediction-index {prediction_index} is out of range for batch size {latent_batch_cpu.shape[0]}."
        )
    latent = latent_batch_cpu[prediction_index:prediction_index + 1].to(device)
    decoded = model.decode_first_stage(latent)
    return to_uint8_image(decoded[0].detach().cpu())


def titled(image, title):
    title_h = 34
    canvas = Image.new("RGB", (image.width, image.height + title_h), "white")
    canvas.paste(image, (0, title_h))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    draw.text((10, 10), title, fill=(0, 0, 0), font=font)
    return canvas


def save_grid(frames, output_path):
    titled_frames = [
        titled(frames["frame0_noise"], "Noise"),
        titled(frames["frame1_after_chunk1"], "After Chunk 1"),
        titled(frames["frame2_after_chunk2"], "After Chunk 2"),
        titled(frames["frame3_final_after_chunk3"], "Final"),
    ]
    w, h = titled_frames[0].size
    grid = Image.new("RGB", (2 * w, 2 * h), "white")
    for idx, frame in enumerate(titled_frames):
        x = (idx % 2) * w
        y = (idx // 2) * h
        grid.paste(frame, (x, y))
    grid.save(output_path)


def write_metadata(path, args, routing_state, config, captures, paths, files, final_latent_max_abs_diff):
    metadata = {
        "sample_index": args.sample_index,
        "prediction_index": args.prediction_index,
        "routing_mode": args.routing_mode,
        "total_sampling_steps": config.ddim_steps,
        "capture_steps": captures,
        "chunk_boundaries": routing_state.bounds,
        "chunk_lengths": [end - start for start, end in routing_state.bounds] if routing_state.bounds else None,
        "stage1_weights": stage_weights("forward", 0.1),
        "stage2_weights": stage_weights("forward", 0.5),
        "stage3_weights": stage_weights("forward", 0.9),
        "stage_weights": routing_weight_summary("forward"),
        "config": paths["config_path"],
        "checkpoint": paths["checkpoint_path"],
        "seed": args.seed,
        "match_forward_rng_order": args.match_forward_rng_order,
        "rng_batches_skipped_before_sample": args.sample_index if args.match_forward_rng_order else 0,
        "sampler_name": "PLMS",
        "num_samples": config.num_samples,
        "output_files": files,
        "debug_shapes": routing_state.debug_shapes,
        "routing_stats": routing_state.token_prior_stats(),
        "progress_stats": routing_state.progress_stats(),
        "cfg_handling": routing_state.cfg_mode,
        "final_latent_max_abs_diff_vs_sampler_return": final_latent_max_abs_diff,
        "notes": (
            "Frames are decoded from latents captured during one real PLMS sampling run. "
            "Frame 0 decodes the initial x_T noise latent; the intermediate frames decode the current latent "
            "after the first and second thirds of denoising. The routing implementation is imported from "
            "code/run_fixed_chunking.py and uses forward routing on post-projector U-Net context tokens. "
            "By default, the script advances the RNG by sample_index initial-noise batches so sample N matches "
            "the normal run_fixed_chunking.py dataset order."
        ),
    }
    path.write_text(json.dumps(metadata, indent=2))


def main():
    import torch
    from einops import repeat

    args = parse_args()
    root = args.root.resolve()

    routing_state = RoutingState("forward", args.eps)
    model_wrap, dataset_test, config, state, paths = build_model_and_data(args, routing_state)
    config.num_samples = args.num_samples
    config.ddim_steps = args.num_sampling_steps

    if args.sample_index < 0 or args.sample_index >= len(dataset_test):
        raise IndexError(f"sample-index {args.sample_index} is out of range for dataset length {len(dataset_test)}.")

    timestamp = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    output_root = resolve_path(root, args.output_dir)
    output_dir = output_root / f"sample{args.sample_index}_seed{args.seed}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)

    device = model_wrap.device
    model = model_wrap.model.to(device)
    model.eval()
    shape = latent_shape(model_wrap, config)
    captures = capture_steps(config.ddim_steps)

    print(f"total steps: {config.ddim_steps}", flush=True)
    print(f"capture steps: {captures}", flush=True)
    print(f"output dir: {output_dir}", flush=True)

    restore_rng_state(state, device)

    item = dataset_test[args.sample_index]
    latent = item["eeg"]
    sampler = ProgressRoutingPLMSSampler(model, routing_state, captures)

    with torch.no_grad(), model.ema_scope(), patch_cross_attention(routing_state, debug_shapes=args.debug_shapes):
        c, re_latent = model.get_learned_conditioning(
            repeat(latent, "h w -> c h w", c=config.num_samples).to(device)
        )
        record_original_conditioner_shapes(model.cond_stage_model, c, re_latent, routing_state)
        routing_state.set_token_count(c.shape[1], c.device, c.dtype)
        routing_state.conditional_batch = c.shape[0]
        if args.match_forward_rng_order and args.sample_index > 0:
            for _ in range(args.sample_index):
                torch.randn((config.num_samples, *shape), device=device)
            print(f"advanced RNG by {args.sample_index} initial-noise batches", flush=True)
        x_T = torch.randn((config.num_samples, *shape), device=device)
        samples_ddim, _ = sampler.sample(
            S=config.ddim_steps,
            x_T=x_T,
            conditioning=c,
            batch_size=config.num_samples,
            shape=shape,
            verbose=False,
        )

        missing = [name for name, step in captures.items() if step not in sampler.latents]
        if missing:
            raise RuntimeError(f"Missing captured latents for frames: {missing}")

        final_diff = float((sampler.latents[config.ddim_steps].to(device) - samples_ddim).abs().max().item())
        frames = {
            name: decode_frame(model, sampler.latents[step], args.prediction_index, device)
            for name, step in captures.items()
        }

    saved_files = []
    frame_names = {
        "frame0_noise": "frame0_noise.png",
        "frame1_after_chunk1": "frame1_after_chunk1.png",
        "frame2_after_chunk2": "frame2_after_chunk2.png",
        "frame3_final_after_chunk3": "frame3_final_after_chunk3.png",
    }
    if args.save_individual_frames:
        for key, filename in frame_names.items():
            path = output_dir / filename
            frames[key].save(path)
            saved_files.append(str(path))

    if args.save_progress_grid:
        grid_path = output_dir / "progress_grid.png"
        save_grid(frames, grid_path)
        saved_files.append(str(grid_path))

    metadata_path = output_dir / "metadata.json"
    write_metadata(metadata_path, args, routing_state, config, captures, paths, saved_files, final_diff)
    saved_files.append(str(metadata_path))

    print(f"chunk boundaries: {routing_state.bounds}", flush=True)
    print(f"chunk lengths: {[end - start for start, end in routing_state.bounds]}", flush=True)
    print(f"stage weights: {routing_weight_summary('forward')}", flush=True)
    print(f"final latent max abs diff vs sampler return: {final_diff:.8g}", flush=True)
    print("saved files:", flush=True)
    for file in saved_files:
        print(f"  {file}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", flush=True)
        raise SystemExit(1)
