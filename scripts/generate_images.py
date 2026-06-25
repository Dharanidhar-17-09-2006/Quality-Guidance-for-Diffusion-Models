
import argparse
import csv
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from core.generation.sd_generator import load_sd_pipeline, generate_image


DEFAULT_PROMPTS = {
    "animals": [
        "A golden retriever sitting on a wooden porch",
        "A tabby cat sleeping on a windowsill",
        "Two horses grazing in an open field",
        "A robin perched on a snow-covered branch",
    ],
    "nature": [
        "A mountain lake surrounded by pine trees on a cloudy afternoon",
        "Rolling green hills with a narrow dirt path winding through them",
        "A sandy beach at sunrise with gentle waves washing ashore",
        "A dense forest with sunlight filtering through the canopy",
    ],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-prompts-per-category", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    output_dir = args.output_dir or (
        args.project / "outputs" / "images" / f"sd_g{args.guidance_scale}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = output_dir / "metadata.csv"

    pipe = load_sd_pipeline(
        project=args.project,
        model_path=args.model_path,
    )

    rows = []

    for category, prompts in DEFAULT_PROMPTS.items():
        selected_prompts = prompts
        if args.max_prompts_per_category is not None:
            selected_prompts = prompts[: args.max_prompts_per_category]

        for i, prompt in enumerate(selected_prompts):
            seed = args.seed + len(rows)
            image_path = output_dir / category / f"image_{i}.png"

            generate_image(
                pipe=pipe,
                prompt=prompt,
                output_path=image_path,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
                seed=seed,
            )

            rows.append({
                "category": category,
                "image": image_path.name,
                "image_path": str(image_path),
                "prompt": prompt,
                "guidance_scale": args.guidance_scale,
                "num_inference_steps": args.num_inference_steps,
                "seed": seed,
            })

            print("Saved:", image_path)

    with metadata_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "category",
                "image",
                "image_path",
                "prompt",
                "guidance_scale",
                "num_inference_steps",
                "seed",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("Metadata saved:", metadata_path)


if __name__ == "__main__":
    main()
