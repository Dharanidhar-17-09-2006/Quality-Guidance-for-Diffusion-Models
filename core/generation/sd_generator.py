
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler


def resolve_sd_model_path(project: Path, model_path=None):
    if model_path is not None:
        return str(model_path)

    candidates = [
        project / "checkpoints" / "sd_v1_5",
        project / "models" / "sd_v1_5",
        Path("/content/drive/MyDrive/diffusion_project/sd_v1_5"),
    ]

    for path in candidates:
        if path.exists():
            return str(path)

    return "runwayml/stable-diffusion-v1-5"


def load_sd_pipeline(project: Path, model_path=None, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    resolved_model_path = resolve_sd_model_path(project, model_path)

    dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = StableDiffusionPipeline.from_pretrained(
        resolved_model_path,
        torch_dtype=dtype,
    )

    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)

    return pipe


def generate_image(
    pipe,
    prompt: str,
    output_path: Path,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 50,
    seed: int = 1234,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = pipe.device.type

    generator = torch.Generator(device=device).manual_seed(seed)

    image = pipe(
        prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=generator,
    ).images[0]

    image.save(output_path)

    return output_path
