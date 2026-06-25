# generate_with_decay_guidance.py
import argparse, csv, sys
from pathlib import Path
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from core.generation.sd_generator import load_sd_pipeline
from core.models.joint_model import JointModel
from scripts.generate_with_without_lfm import LFMScorer


def get_vae_scale_factor(pipe):
    if hasattr(pipe, "vae_scale_factor"):
        return pipe.vae_scale_factor
    return 2 ** (len(pipe.vae.config.block_out_channels) - 1)


def encode_prompt(pipe, prompt, device):
    text_inputs = pipe.tokenizer(
        [prompt], padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True, return_tensors="pt",
    )
    uncond_inputs = pipe.tokenizer(
        [""], padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        return_tensors="pt",
    )
    with torch.no_grad():
        text_embeds   = pipe.text_encoder(text_inputs.input_ids.to(device))[0]
        uncond_embeds = pipe.text_encoder(uncond_inputs.input_ids.to(device))[0]
    return torch.cat([uncond_embeds, text_embeds], dim=0)


def decode_latents(pipe, latents):
    latents = latents / pipe.vae.config.scaling_factor
    image   = pipe.vae.decode(latents).sample
    return (image / 2 + 0.5).clamp(0, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project",              type=Path,  required=True)
    parser.add_argument("--prompt",               type=str,   required=True)
    parser.add_argument("--output-dir",           type=Path,  default=None)
    parser.add_argument("--model-path",           type=str,   default=None)
    parser.add_argument("--num-inference-steps",  type=int,   default=30)
    parser.add_argument("--guidance-scale",       type=float, default=7.5)
    parser.add_argument("--lfm-method",           type=str,   default="gaussian",
                        choices=["none", "gaussian", "at"])
    parser.add_argument("--min-denoise-weight",   type=float, default=0.5)
    parser.add_argument("--rat-weight-start",     type=float, default=0.1)
    parser.add_argument("--rat-weight-end",       type=float, default=1.0)
    parser.add_argument("--guidance-last-steps",  type=int,   default=10)
    parser.add_argument("--post-guidance-steps",  type=int,   default=0)
    parser.add_argument("--post-guidance-lr",     type=float, default=0.005)
    parser.add_argument("--seed",                 type=int,   default=42)
    args = parser.parse_args()

    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = args.output_dir or (args.project / "outputs" / "decay_guidance")
    output_dir.mkdir(parents=True, exist_ok=True)

    pipe = load_sd_pipeline(project=args.project, model_path=args.model_path, device=device.type)
    if hasattr(pipe, "safety_checker"):
        pipe.safety_checker = None

    joint_model = JointModel(use_pretrained_backbones=True).to(device)
    state = torch.load(args.project / "checkpoints" / "JOINT_2024.pth", map_location=device)
    joint_model.load_state_dict(state)
    joint_model.eval()

    scorer = LFMScorer(joint_model, lfm_method=args.lfm_method).to(device)
    scorer.eval()

    default_size = pipe.unet.config.sample_size
    latent_size  = default_size if isinstance(default_size, int) else default_size[0]

    pipe.scheduler.set_timesteps(args.num_inference_steps, device=device)
    timesteps      = pipe.scheduler.timesteps
    alphas_cumprod = pipe.scheduler.alphas_cumprod.to(device)
    guidance_start = max(0, len(timesteps) - args.guidance_last_steps)
    guided_count   = len(timesteps) - guidance_start

    prompt_embeds = encode_prompt(pipe, args.prompt, device)
    generator     = torch.Generator(device=device).manual_seed(args.seed)
    dtype         = pipe.unet.dtype

    latents = torch.randn(
        (1, pipe.unet.config.in_channels, latent_size, latent_size),
        generator=generator, device=device, dtype=dtype,
    ) * pipe.scheduler.init_noise_sigma

    print("Starting denoising...")

    for i, t in enumerate(timesteps):
        apply_guidance = i >= guidance_start

        if apply_guidance:
            progress       = (i - guidance_start) / max(guided_count - 1, 1)
            denoise_weight = 1.0 - progress * (1.0 - args.min_denoise_weight)
            rat_weight     = args.rat_weight_start + progress * (
                args.rat_weight_end - args.rat_weight_start)

        with torch.set_grad_enabled(apply_guidance):
            if apply_guidance:
                latents = latents.detach().requires_grad_(True)

            latent_in  = pipe.scheduler.scale_model_input(torch.cat([latents]*2), t)
            noise_pred = pipe.unet(latent_in, t, encoder_hidden_states=prompt_embeds).sample
            noise_uncond, noise_text = noise_pred.chunk(2)
            noise_cfg = noise_uncond + args.guidance_scale * (noise_text - noise_uncond)

            if apply_guidance:
                alpha_bar = alphas_cumprod[t.long()].to(dtype=latents.dtype)
                x0_pred   = (latents - (1-alpha_bar).sqrt() * noise_cfg) / alpha_bar.sqrt()
                x0_image  = decode_latents(pipe, x0_pred)
                score     = scorer(x0_image.float()).sum()
                grad      = torch.autograd.grad(score, latents)[0]

                if torch.isnan(grad).any() or torch.isinf(grad).any():
                    grad = torch.zeros_like(grad)
                else:
                    grad_f32  = grad.float()
                    grad_norm = grad_f32.flatten(1).norm(dim=1).view(-1,1,1,1)
                    grad      = (grad_f32 / (grad_norm + 1e-8)).to(latents.dtype)

                noise_cfg = noise_cfg - rat_weight * (1-alpha_bar).sqrt() * grad
                print(f"[step {i}] denoise_w={denoise_weight:.3f} rat_w={rat_weight:.3f} score={score.item():.4f}")

                noise_cfg = noise_cfg.detach()
                latents   = latents.detach()

        latents = pipe.scheduler.step(noise_cfg, t, latents).prev_sample

    # post-denoising gradient ascent
    if args.post_guidance_steps > 0:
        print(f"\nPost-denoising ascent ({args.post_guidance_steps} steps)...")
        latents = latents.detach().requires_grad_(True)

        for pg in range(args.post_guidance_steps):
            x0_image = decode_latents(pipe, latents)
            score    = scorer(x0_image.float()).sum()
            grad     = torch.autograd.grad(score, latents)[0]

            if torch.isnan(grad).any() or torch.isinf(grad).any():
                print(f"  post step {pg+1} | NaN grad — stopping")
                break

            grad_f32  = grad.float()
            grad_norm = grad_f32.flatten(1).norm(dim=1).view(-1,1,1,1)
            grad      = (grad_f32 / (grad_norm + 1e-8)).to(latents.dtype)

            latents = (latents + args.post_guidance_lr * grad).detach().requires_grad_(True)
            print(f"  post step {pg+1} | score={score.item():.4f}")

    with torch.no_grad():
        final_image = decode_latents(pipe, latents.detach())

    image_np = final_image.detach().cpu().permute(0,2,3,1).float().numpy()
    images   = pipe.numpy_to_pil(image_np)
    out_path = output_dir / "guided_000.png"
    images[0].save(out_path)
    print("Saved:", out_path)

    with open(output_dir / "metadata.csv", "w", newline="") as f:
        import csv
        writer = csv.DictWriter(f, fieldnames=["prompt","image_path"])
        writer.writeheader()
        writer.writerow({"prompt": args.prompt, "image_path": str(out_path)})


if __name__ == "__main__":
    main()