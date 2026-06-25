
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
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--resize", type=int, default=384)
    parser.add_argument("--crop-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    image_dir = args.image_dir
    image_files = sorted(
        p for p in image_dir.rglob("*")
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
    )

    if not image_files:
        raise FileNotFoundError(f"No images found under {image_dir}")

    output_csv = args.output_csv or (
        args.project / "outputs" / "scores" / f"{image_dir.name}_own_joint_scores.csv"
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    patch_root = args.project / "outputs" / "patch_partitioned_generated" / image_dir.name
    patch_root.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print("Image dir:", image_dir)
    print("Found images:", len(image_files))
    print("Output CSV:", output_csv)

    model_path = args.project / "checkpoints" / "JOINT_2024.pth"
    pal_path = args.project / "checkpoints" / "end2end.pt"

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

    for idx, image_path in enumerate(image_files):
        rel = image_path.relative_to(image_dir)
        technical_path = patch_root / rel
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
            project=args.project,
            device=device,
        )

        row = {
            "image": rel.as_posix(),
            "image_path": str(image_path),
            "technical_path": str(technical_path),
            **scores,
        }
        rows.append(row)

        print(
            f"{idx + 1}/{len(image_files)} {rel} "
            f"T={scores['technical']:.4f} "
            f"R={scores['rationality']:.4f} "
            f"N={scores['naturalness']:.4f}"
        )

    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image",
                "image_path",
                "technical_path",
                "technical",
                "rationality",
                "naturalness",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("Saved:", output_csv)


if __name__ == "__main__":
    main()
