import argparse
import csv
import sys
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from core.generation.sd_generator import load_sd_pipeline
from core.models.joint_model import JointModel
from core.guidance.rationality_guidance import RationalityGuidanceScorer


def get_vae_scale_factor(pipe):
    if hasattr(pipe, "vae_scale_factor"):
        return pipe.vae_scale_factor
    return 2 ** (len(pipe.vae.config.block_out_channels) - 1)


def encode_prompt(pipe, prompt, batch_size, device):
    text_inputs = pipe.tokenizer(
        [prompt],
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )

    uncond_inputs = pipe.tokenizer(
        [""],
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        return_tensors="pt",
    )

    with torch.no_grad():
        text_embeds = pipe.text_encoder(
            text_inputs.input_ids.to(device)
        )[0]

        uncond_embeds = pipe.text_encoder(
            uncond_inputs.input_ids.to(device)
        )[0]

    text_embeds = text_embeds.repeat(batch_size, 1, 1)
    uncond_embeds = uncond_embeds.repeat(batch_size, 1, 1)

    return torch.cat([uncond_embeds, text_embeds], dim=0)


def decode_latents(pipe, latents):
    latents = latents / pipe.vae.config.scaling_factor

    image = pipe.vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)

    return image


def save_images(pipe, image_tensor, output_dir, rows, prompt):
    image_np = (
        image_tensor.detach()
        .cpu()
        .permute(0, 2, 3, 1)
        .float()
        .numpy()
    )

    images = pipe.numpy_to_pil(image_np)

    for i, img in enumerate(images):
        out = output_dir / f"guided_{i:03d}.png"

        img.save(out)

        rows.append({
            "index": i,
            "prompt": prompt,
            "image_path": str(out),
        })

        print("Saved:", out)


def save_debug_image(pipe, image_tensor, save_path):
    image_np = (
        image_tensor[0]
        .detach()
        .cpu()
        .permute(1, 2, 0)
        .float()
        .numpy()
    )

    image = pipe.numpy_to_pil([image_np])[0]
    image.save(save_path)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--prompt", type=str, required=True)

    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)

    parser.add_argument("--num-images", type=int, default=1)

    parser.add_argument("--num-inference-steps", type=int, default=30)

    parser.add_argument("--guidance-scale", type=float, default=7.5)

    parser.add_argument(
        "--rationality-weight",
        type=float,
        default=0.0005,
    )

    parser.add_argument(
        "--guidance-last-steps",
        type=int,
        default=5,
    )

    parser.add_argument("--seed", type=int, default=1234)

    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)

    parser.add_argument(
        "--save-debug-steps",
        action="store_true",
    )

    args = parser.parse_args()

    project = args.project

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )

    print("Using device:", device)

    output_dir = args.output_dir or (
        project / "outputs" / "images" / "rationality_guided"
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    debug_dir = output_dir / "debug_steps"
    debug_dir.mkdir(parents=True, exist_ok=True)

    pipe = load_sd_pipeline(
        project=project,
        model_path=args.model_path,
        device=device.type,
    )

    if hasattr(pipe, "safety_checker"):
        pipe.safety_checker = None

    vae_scale = get_vae_scale_factor(pipe)

    default_latent_size = pipe.unet.config.sample_size

    if isinstance(default_latent_size, tuple):
        latent_h, latent_w = default_latent_size
    else:
        latent_h = latent_w = default_latent_size

    height = args.height or latent_h * vae_scale
    width = args.width or latent_w * vae_scale

    latent_h = height // vae_scale
    latent_w = width // vae_scale

    print("Loading JOINT model...")

    joint_model = JointModel(
        use_pretrained_backbones=True
    ).to(device)

    state = torch.load(
        project / "checkpoints" / "JOINT_2024.pth",
        map_location=device,
    )

    joint_model.load_state_dict(state)
    joint_model.eval()

    print("JOINT loaded")

    rationality_scorer = RationalityGuidanceScorer(
        joint_model
    ).to(device)

    rationality_scorer.eval()

    pipe.scheduler.set_timesteps(
        args.num_inference_steps,
        device=device,
    )

    timesteps = pipe.scheduler.timesteps

    alphas_cumprod = (
        pipe.scheduler.alphas_cumprod.to(device)
    )

    guidance_start = max(
        0,
        len(timesteps) - args.guidance_last_steps,
    )

    print("Guidance starts at step:", guidance_start)

    prompt_embeds = encode_prompt(
        pipe=pipe,
        prompt=args.prompt,
        batch_size=args.num_images,
        device=device,
    )

    generator = torch.Generator(
        device=device
    ).manual_seed(args.seed)

    dtype = pipe.unet.dtype

    latents = torch.randn(
        (
            args.num_images,
            pipe.unet.config.in_channels,
            latent_h,
            latent_w,
        ),
        generator=generator,
        device=device,
        dtype=dtype,
    )

    latents = latents * pipe.scheduler.init_noise_sigma

    print("\nStarting denoising...\n")

    for i, t in enumerate(timesteps):

        apply_guidance = i >= guidance_start

        with torch.set_grad_enabled(apply_guidance):

            if apply_guidance:
                latents = latents.detach().requires_grad_(True)

            latent_model_input = torch.cat(
                [latents] * 2,
                dim=0,
            )

            latent_model_input = pipe.scheduler.scale_model_input(
                latent_model_input,
                t,
            )

            noise_pred = pipe.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
            ).sample

            noise_uncond, noise_text = noise_pred.chunk(2)

            noise_cfg = (
                noise_uncond
                + args.guidance_scale
                * (noise_text - noise_uncond)
            )

            if apply_guidance:

                noise_cfg_original = noise_cfg.clone()

                alpha_bar = alphas_cumprod[
                    t.long()
                ].to(
                    device=device,
                    dtype=latents.dtype,
                )

                x0_pred = (
                    latents
                    - torch.sqrt(1 - alpha_bar)
                    * noise_cfg
                ) / torch.sqrt(alpha_bar)

                x0_image = decode_latents(
                    pipe,
                    x0_pred,
                )

                score = rationality_scorer(
                    x0_image.float()
                ).sum()

                score_before = score.item()

                grad = torch.autograd.grad(
                    score,
                    latents,
                )[0]

                print(
                    f"[step {i}] "
                    f"score={score_before:.6f} "
                    f"grad_mean={grad.abs().mean().item():.6f} "
                    f"grad_norm={grad.norm().item():.6f}"
                )

                if (
                    torch.isnan(grad).any()
                    or torch.isinf(grad).any()
                ):
                    print(
                        f"[step {i}] "
                        f"WARNING: NaN/Inf gradients"
                    )

                    grad = torch.zeros_like(grad)

                else:
                    grad_f32 = grad.float()

                    grad_norm = (
                        grad_f32.flatten(1)
                        .norm(dim=1)
                        .view(-1, 1, 1, 1)
                    )

                    grad = (
                        grad_f32
                        / (grad_norm + 1e-8)
                    ).to(latents.dtype)

                    # optional clipping
                    # comment this out if needed
                    grad = torch.clamp(
                        grad,
                        -0.1,
                        0.1,
                    )

                t_scale = torch.sqrt(1 - alpha_bar)

                adaptive_weight = (
                    args.rationality_weight
                    * t_scale
                    * 5.0
                )

                noise_cfg = noise_cfg - (
                    adaptive_weight * grad
                )

                trajectory_shift = (
                    noise_cfg - noise_cfg_original
                ).abs().mean().item()

                print(
                    f"[step {i}] "
                    f"trajectory_shift={trajectory_shift:.10f}"
                )

                with torch.no_grad():

                    x0_after = (
                        latents
                        - torch.sqrt(1 - alpha_bar)
                        * noise_cfg
                    ) / torch.sqrt(alpha_bar)

                    x0_after_image = decode_latents(
                        pipe,
                        x0_after,
                    )

                    score_after = (
                        rationality_scorer(
                            x0_after_image.float()
                        )
                        .mean()
                        .item()
                    )

                print(
                    f"[step {i}] "
                    f"score_before={score_before:.6f} "
                    f"score_after={score_after:.6f}"
                )

                noise_cfg = noise_cfg.detach()
                latents = latents.detach()

                # save debug images every few steps
                if (
                    args.save_debug_steps
                    and i % 5 == 0
                ):
                    debug_path = (
                        debug_dir
                        / f"step_{i:03d}.png"
                    )

                    save_debug_image(
                        pipe,
                        x0_image,
                        debug_path,
                    )

                    print(
                        f"[step {i}] "
                        f"saved debug image"
                    )

        latents = pipe.scheduler.step(
            noise_cfg,
            t,
            latents,
        ).prev_sample

        print(
            f"step {i + 1}/{len(timesteps)} "
            f"guided={apply_guidance}"
        )

    print("\nDecoding final images...\n")

    with torch.no_grad():
        final_images = decode_latents(
            pipe,
            latents,
        )

    rows = []

    save_images(
        pipe,
        final_images,
        output_dir,
        rows,
        args.prompt,
    )

    metadata = output_dir / "metadata.csv"

    with metadata.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "index",
                "prompt",
                "image_path",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    print("\nMetadata:", metadata)
    print("\nDONE")


if __name__ == "__main__":
    main()