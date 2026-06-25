import argparse
import csv
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from core.generation.sd_generator import load_sd_pipeline
from core.models.joint_model import JointModel


# ──────────────────────────────────────────────
# LFM preprocessing methods
# ──────────────────────────────────────────────

def _imagenet_normalize(x):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
    return (x - mean) / std


def _resize(x, size=224):
    return F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)


def preprocess_none(x):
    """No LFM — pass raw image directly."""
    return _imagenet_normalize(_resize(x))


def _gaussian_kernel_2d(k, sigma):
    half = k // 2
    ax = torch.arange(-half, half + 1, dtype=torch.float32)
    k1d = torch.exp(-0.5 * (ax / sigma) ** 2)
    k1d = k1d / k1d.sum()
    return torch.outer(k1d, k1d)


def preprocess_gaussian(x, blur_kernel=11, blur_sigma=None):
    """Gaussian blur as LFM approximation."""
    if blur_sigma is None:
        blur_sigma = 0.3 * ((blur_kernel - 1) / 2 - 1) + 0.8
    kernel = _gaussian_kernel_2d(blur_kernel, blur_sigma)
    kernel = kernel.unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1).to(x.device)
    pad = blur_kernel // 2
    x = F.conv2d(x, kernel.to(dtype=x.dtype), padding=pad, groups=3)
    return _imagenet_normalize(_resize(x))


class ATMinimizer(nn.Module):
    """Differentiable PyTorch reimplementation of AmbrosioTortorelli minimizer."""

    def __init__(self, iterations=3, solver_maxiterations=6,
                 tol=0.1, alpha=1000.0, beta=0.01, epsilon=0.01):
        super().__init__()
        self.iterations = iterations
        self.solver_maxiterations = solver_maxiterations
        self.tol = tol
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.register_buffer('k_x',   torch.tensor([[[-1., 0., 1.]]]).view(1, 1, 1, 3))
        self.register_buffer('k_y',   torch.tensor([[[-1.], [0.], [1.]]]).view(1, 1, 3, 1))
        self.register_buffer('k_lap', torch.tensor([[[0., 1., 0.],
                                                      [1., -4., 1.],
                                                      [0., 1., 0.]]]).view(1, 1, 3, 3))

    def _gradients(self, x):
        return (F.conv2d(x, self.k_x, padding=(0, 1)),
                F.conv2d(x, self.k_y, padding=(1, 0)))

    def _cg_solve(self, A_func, b, x0):
        x = x0.clone()
        r = b - A_func(x0)
        p = r.clone()
        rsold = (r * r).sum()
        for _ in range(self.solver_maxiterations):
            Ap    = A_func(p)
            alpha = rsold / ((p * Ap).sum() + 1e-8)
            x     = x + alpha * p
            r     = r - alpha * Ap
            rsnew = (r * r).sum()
            if torch.sqrt(rsnew) < self.tol:
                break
            p     = r + (rsnew / rsold) * p
            rsold = rsnew
        return x

    def forward(self, img_channel):
        g  = img_channel / (img_channel.max() + 1e-8)
        f  = g.clone()
        edges = torch.zeros_like(g)
        add_const  = self.beta / (4 * self.epsilon)
        mult_const = self.epsilon * self.beta

        for _ in range(self.iterations):
            gx, gy  = self._gradients(f)
            grad_mag = gx ** 2 + gy ** 2

            def edge_op(v):
                return (v * (grad_mag * self.alpha + add_const)
                        - mult_const * F.conv2d(v, self.k_lap, padding=1))

            edges    = self._cg_solve(edge_op, torch.ones_like(g) * add_const, edges)
            esq      = edges ** 2

            def img_op(f_in):
                fx, fy = self._gradients(f_in)
                tx = F.conv2d(esq * fx, self.k_x,   padding=(0, 1))
                ty = F.conv2d(esq * fy, self.k_y,   padding=(1, 0))
                return f_in - 2 * self.alpha * (tx + ty)

            f = self._cg_solve(img_op, g, f)

        f_min, f_max = f.min(), f.max()
        return (f - f_min) / (f_max - f_min + 1e-8)


def preprocess_at(x, at_minimizer):
    """Full AT minimizer as LFM."""
    channels = torch.unbind(x, dim=1)
    lfm = torch.cat([at_minimizer(ch.unsqueeze(1)) for ch in channels], dim=1)
    return _imagenet_normalize(_resize(lfm))


# ──────────────────────────────────────────────
# Scorer wrapper
# ──────────────────────────────────────────────

class LFMScorer(nn.Module):
    def __init__(self, joint_model, lfm_method="gaussian", blur_kernel=11):
        super().__init__()
        self.joint_model  = joint_model
        self.lfm_method   = lfm_method
        self.blur_kernel  = blur_kernel

        for p in self.joint_model.parameters():
            p.requires_grad_(False)
        self.joint_model.eval()

        if lfm_method == "at":
            self.at_minimizer = ATMinimizer()
        else:
            self.at_minimizer = None

    def forward(self, x):
        if self.lfm_method == "none":
            x = preprocess_none(x)
        elif self.lfm_method == "gaussian":
            x = preprocess_gaussian(x, blur_kernel=self.blur_kernel)
        elif self.lfm_method == "at":
            x = preprocess_at(x, self.at_minimizer.to(x.device))
        features = self.joint_model.rationality_feature_extraction(x)
        return self.joint_model.quality_R(features)


# ──────────────────────────────────────────────
# Pipeline helpers
# ──────────────────────────────────────────────

def get_vae_scale_factor(pipe):
    if hasattr(pipe, "vae_scale_factor"):
        return pipe.vae_scale_factor
    return 2 ** (len(pipe.vae.config.block_out_channels) - 1)


def encode_prompt(pipe, prompt, batch_size, device):
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
        text_embeds  = pipe.text_encoder(text_inputs.input_ids.to(device))[0]
        uncond_embeds = pipe.text_encoder(uncond_inputs.input_ids.to(device))[0]
    text_embeds   = text_embeds.repeat(batch_size, 1, 1)
    uncond_embeds = uncond_embeds.repeat(batch_size, 1, 1)
    return torch.cat([uncond_embeds, text_embeds], dim=0)


def decode_latents(pipe, latents):
    latents = latents / pipe.vae.config.scaling_factor
    image   = pipe.vae.decode(latents).sample
    return (image / 2 + 0.5).clamp(0, 1)


def save_images(pipe, image_tensor, output_dir, rows, prompt):
    image_np = image_tensor.detach().cpu().permute(0, 2, 3, 1).float().numpy()
    images   = pipe.numpy_to_pil(image_np)
    for i, img in enumerate(images):
        out = output_dir / f"guided_{i:03d}.png"
        img.save(out)
        rows.append({"index": i, "prompt": prompt, "image_path": str(out)})
        print("Saved:", out)


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project",              type=Path,  required=True)
    parser.add_argument("--prompt",               type=str,   required=True)
    parser.add_argument("--lfm-method",           type=str,   default="gaussian",
                        choices=["none", "gaussian", "at"],
                        help="LFM preprocessing: none | gaussian | at")
    parser.add_argument("--model-path",           type=str,   default=None)
    parser.add_argument("--output-dir",           type=Path,  default=None)
    parser.add_argument("--num-images",           type=int,   default=1)
    parser.add_argument("--num-inference-steps",  type=int,   default=30)
    parser.add_argument("--guidance-scale",       type=float, default=7.5)
    parser.add_argument("--rationality-weight",   type=float, default=0.5)
    parser.add_argument("--guidance-last-steps",  type=int,   default=10)
    parser.add_argument("--blur-kernel",          type=int,   default=11)
    parser.add_argument("--seed",                 type=int,   default=42)
    parser.add_argument("--height",               type=int,   default=None)
    parser.add_argument("--width",                type=int,   default=None)
    args = parser.parse_args()

    project = args.project
    device  = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("LFM method:", args.lfm_method)

    output_dir = args.output_dir or (
        project / "outputs" / "lfm_comparison" / args.lfm_method
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    pipe = load_sd_pipeline(project=project, model_path=args.model_path, device=device.type)
    if hasattr(pipe, "safety_checker"):
        pipe.safety_checker = None

    vae_scale = get_vae_scale_factor(pipe)
    default_latent_size = pipe.unet.config.sample_size
    if isinstance(default_latent_size, tuple):
        latent_h, latent_w = default_latent_size
    else:
        latent_h = latent_w = default_latent_size

    height   = args.height or latent_h * vae_scale
    width    = args.width  or latent_w * vae_scale
    latent_h = height // vae_scale
    latent_w = width  // vae_scale

    print("Loading JOINT model...")
    joint_model = JointModel(use_pretrained_backbones=True).to(device)
    state = torch.load(project / "checkpoints" / "JOINT_2024.pth", map_location=device)
    joint_model.load_state_dict(state)
    joint_model.eval()

    scorer = LFMScorer(
        joint_model,
        lfm_method=args.lfm_method,
        blur_kernel=args.blur_kernel,
    ).to(device)
    scorer.eval()

    pipe.scheduler.set_timesteps(args.num_inference_steps, device=device)
    timesteps      = pipe.scheduler.timesteps
    alphas_cumprod = pipe.scheduler.alphas_cumprod.to(device)
    guidance_start = max(0, len(timesteps) - args.guidance_last_steps)

    prompt_embeds = encode_prompt(pipe, args.prompt, args.num_images, device)

    generator = torch.Generator(device=device).manual_seed(args.seed)
    dtype     = pipe.unet.dtype

    latents = torch.randn(
        (args.num_images, pipe.unet.config.in_channels, latent_h, latent_w),
        generator=generator, device=device, dtype=dtype,
    ) * pipe.scheduler.init_noise_sigma

    print("\nStarting denoising...\n")

    for i, t in enumerate(timesteps):
        apply_guidance = i >= guidance_start

        with torch.set_grad_enabled(apply_guidance):
            if apply_guidance:
                latents = latents.detach().requires_grad_(True)

            latent_in = torch.cat([latents] * 2)
            latent_in = pipe.scheduler.scale_model_input(latent_in, t)

            noise_pred = pipe.unet(
                latent_in, t, encoder_hidden_states=prompt_embeds
            ).sample

            noise_uncond, noise_text = noise_pred.chunk(2)
            noise_cfg = noise_uncond + args.guidance_scale * (noise_text - noise_uncond)

            if apply_guidance:
                alpha_bar = alphas_cumprod[t.long()].to(device=device, dtype=latents.dtype)

                x0_pred  = (latents - torch.sqrt(1 - alpha_bar) * noise_cfg) / torch.sqrt(alpha_bar)
                x0_image = decode_latents(pipe, x0_pred)
                score    = scorer(x0_image.float()).sum()

                grad = torch.autograd.grad(score, latents)[0]

                if torch.isnan(grad).any() or torch.isinf(grad).any():
                    grad = torch.zeros_like(grad)
                else:
                    grad_f32  = grad.float()
                    grad_norm = grad_f32.flatten(1).norm(dim=1).view(-1, 1, 1, 1)
                    grad      = (grad_f32 / (grad_norm + 1e-8)).to(latents.dtype)

                noise_cfg = noise_cfg - (
                    args.rationality_weight * torch.sqrt(1 - alpha_bar) * grad
                )

                print(
                    f"[step {i}] score={score.item():.4f} "
                    f"grad_norm={grad.norm().item():.4f} "
                    f"lfm={args.lfm_method}"
                )

                noise_cfg = noise_cfg.detach()
                latents   = latents.detach()

        latents = pipe.scheduler.step(noise_cfg, t, latents).prev_sample
        print(f"step {i+1}/{len(timesteps)} guided={apply_guidance}")

    with torch.no_grad():
        final_images = decode_latents(pipe, latents)

    rows = []
    save_images(pipe, final_images, output_dir, rows, args.prompt)

    metadata = output_dir / "metadata.csv"
    with metadata.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["index", "prompt", "image_path"])
        writer.writeheader()
        writer.writerows(rows)

    print("\nMetadata:", metadata)
    print("DONE")


if __name__ == "__main__":
    main()