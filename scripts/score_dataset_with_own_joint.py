
import argparse
import csv
import sys
from pathlib import Path

import torch
from PIL import Image

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from core.models.joint_model import JointModel
from core.preprocessing.lfm import compute_lfm_pil
from core.preprocessing.patch_partition import load_pal_model, patch_partition_image
from core.preprocessing.transforms import build_joint_transform


def score_one_image(
    image_path,
    technical_path,
    model,
    transform,
    project,
    device,
):
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


def find_dataset_root(project):
    candidates = [
        project / "data",
        project,
        Path("/content/drive/MyDrive/diffusion_project"),
    ]

    for root in candidates:
        if (root / "no_cfg").exists() and (root / "with_cfg").exists():
            return root

    raise FileNotFoundError(
        "Could not find dataset root containing no_cfg and with_cfg folders."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--max-images-per-scale", type=int, default=None)
    parser.add_argument("--resize", type=int, default=384)
    parser.add_argument("--crop-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    project = args.project
    data_root = args.data_root or find_dataset_root(project)

    output_csv = args.output_csv or (
        project / "outputs" / "scores" / "own_joint_scores.csv"
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    patch_root = project / "outputs" / "patch_partitioned_dataset"
    patch_root.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print("Data root:", data_root)
    print("Output CSV:", output_csv)

    model_path = project / "checkpoints" / "JOINT_2024.pth"
    pal_path = project / "checkpoints" / "end2end.pt"

    model = JointModel(use_pretrained_backbones=True).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    pal_model = load_pal_model(pal_path, device)

    transform = build_joint_transform(
        resize=args.resize,
        crop_size=args.crop_size,
    )

    rows = []

    for scale in ["no_cfg", "with_cfg"]:
        image_files = sorted((data_root / scale).rglob("*.png"))

        if args.max_images_per_scale is not None:
            image_files = image_files[: args.max_images_per_scale]

        print(f"Scoring {len(image_files)} images for {scale}")

        for idx, image_path in enumerate(image_files):
            rel = image_path.relative_to(data_root / scale)
            technical_path = patch_root / scale / rel
            technical_path.parent.mkdir(parents=True, exist_ok=True)

            if not technical_path.exists():
                patch_partition_image(
                    image_path=image_path,
                    output_path=technical_path,
                    pal_model=pal_model,
                    device=device,
                    seed=args.seed,
                )

            scores = score_one_image(
                image_path=image_path,
                technical_path=technical_path,
                model=model,
                transform=transform,
                project=project,
                device=device,
            )

            row = {
                "scale": scale,
                "category": rel.parent.as_posix(),
                "image": rel.name,
                "image_path": str(image_path),
                "technical_path": str(technical_path),
                **scores,
            }
            rows.append(row)

            print(
                f"[{scale}] {idx + 1}/{len(image_files)} {rel} "
                f"T={scores['technical']:.4f} "
                f"R={scores['rationality']:.4f} "
                f"N={scores['naturalness']:.4f}"
            )

    fieldnames = [
        "scale",
        "category",
        "image",
        "image_path",
        "technical_path",
        "technical",
        "rationality",
        "naturalness",
    ]

    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("Saved:", output_csv)

    # Print simple summary without requiring pandas.
    for scale in ["no_cfg", "with_cfg"]:
        subset = [r for r in rows if r["scale"] == scale]
        if not subset:
            continue

        mean_t = sum(r["technical"] for r in subset) / len(subset)
        mean_r = sum(r["rationality"] for r in subset) / len(subset)
        mean_n = sum(r["naturalness"] for r in subset) / len(subset)

        print(
            f"{scale}: "
            f"technical={mean_t:.4f}, "
            f"rationality={mean_r:.4f}, "
            f"naturalness={mean_n:.4f}"
        )


if __name__ == "__main__":
    main()
