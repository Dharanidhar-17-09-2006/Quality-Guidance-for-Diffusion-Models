
import argparse
import csv
import shutil
import sys
from pathlib import Path

import torch
from PIL import Image

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from core.generation.sd_generator import load_sd_pipeline, generate_image
from core.models.joint_model import JointModel
from core.preprocessing.lfm import compute_lfm_pil
from core.preprocessing.patch_partition import load_pal_model, patch_partition_image
from core.preprocessing.transforms import build_joint_transform


def score_one_image(image_path, technical_path, model, transform, project, device):
    original_image = Image.open(image_path).convert("RGB")
    technical_image = Image.open(technical_path).convert("RGB")
    rationality_image = compute_lfm_pil(original_image, project=project)

    x_technical = transform(technical_image).unsqueeze(0).to(device)
    x_rationality = transform(rationality_image).unsqueeze(0).to(device)

    with torch.no_grad():
        score_t, score_r = model(x_technical, x_rationality)
        score_n = model.naturalness_score(score_t, score_r)

    return {
        "technical": float(score_t.item()),
        "rationality": float(score_r.item()),
        "naturalness": float(score_n.item()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--num-candidates", type=int, default=4)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--rank-by", type=str, default="naturalness",
                        choices=["technical", "rationality", "naturalness"])
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    project = args.project
    output_dir = args.output_dir or (
        project / "outputs" / "images" / "generate_and_rank"
    )
    candidates_dir = output_dir / "candidates"
    patches_dir = output_dir / "patch_partitioned"
    best_dir = output_dir / "best"
    scores_csv = output_dir / "scores.csv"

    candidates_dir.mkdir(parents=True, exist_ok=True)
    patches_dir.mkdir(parents=True, exist_ok=True)
    best_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print("Prompt:", args.prompt)

    pipe = load_sd_pipeline(project=project, device=device.type)

    joint_model = JointModel(use_pretrained_backbones=True).to(device)
    state = torch.load(project / "checkpoints" / "JOINT_2024.pth", map_location=device)
    joint_model.load_state_dict(state)
    joint_model.eval()

    pal_model = load_pal_model(project / "checkpoints" / "end2end.pt", device)

    transform = build_joint_transform()

    rows = []

    for i in range(args.num_candidates):
        seed = args.seed + i
        image_path = candidates_dir / f"candidate_{i:03d}.png"
        technical_path = patches_dir / f"candidate_{i:03d}.png"

        generate_image(
            pipe=pipe,
            prompt=args.prompt,
            output_path=image_path,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            seed=seed,
        )

        patch_partition_image(
            image_path=image_path,
            output_path=technical_path,
            pal_model=pal_model,
            device=device,
            seed=seed,
        )

        scores = score_one_image(
            image_path=image_path,
            technical_path=technical_path,
            model=joint_model,
            transform=transform,
            project=project,
            device=device,
        )

        row = {
            "candidate": i,
            "seed": seed,
            "prompt": args.prompt,
            "image_path": str(image_path),
            "technical_path": str(technical_path),
            **scores,
        }
        rows.append(row)

        print(
            f"candidate {i} seed={seed} "
            f"T={scores['technical']:.4f} "
            f"R={scores['rationality']:.4f} "
            f"N={scores['naturalness']:.4f}"
        )

    rows = sorted(rows, key=lambda r: r[args.rank_by], reverse=True)
    best = rows[0]

    best_image_path = best_dir / "best.png"
    shutil.copy2(best["image_path"], best_image_path)

    with scores_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "candidate",
                "seed",
                "prompt",
                "image_path",
                "technical_path",
                "technical",
                "rationality",
                "naturalness",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("Ranked by:", args.rank_by)
    print("Best candidate:", best["candidate"])
    print("Best image:", best_image_path)
    print("Scores CSV:", scores_csv)


if __name__ == "__main__":
    main()
