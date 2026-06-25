
import argparse
import sys
from pathlib import Path

import torch
from PIL import Image

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from core.models.joint_model import JointModel
from core.preprocessing.lfm import compute_lfm_pil
from core.preprocessing.transforms import build_joint_transform


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--technical-image", type=Path, default=None)
    parser.add_argument("--resize", type=int, default=384)
    parser.add_argument("--crop_size", type=int, default=224)
    args = parser.parse_args()

    model_path = args.project / "checkpoints" / "JOINT_2024.pth"
    technical_path = args.technical_image or Path(str(args.image).replace("demo", "AGIN_patch_partitioned"))

    if not args.image.exists():
        raise FileNotFoundError(f"Missing image: {args.image}")
    if not technical_path.exists():
        raise FileNotFoundError(f"Missing technical image: {technical_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = JointModel(use_pretrained_backbones=True).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    transform = build_joint_transform(
        resize=args.resize,
        crop_size=args.crop_size,
    )

    original_image = Image.open(args.image).convert("RGB")
    technical_image = Image.open(technical_path).convert("RGB")
    rationality_image = compute_lfm_pil(original_image, project=args.project)

    x_technical = transform(technical_image).unsqueeze(0).to(device)
    x_rationality = transform(rationality_image).unsqueeze(0).to(device)

    with torch.no_grad():
        score_T, score_R = model(x_technical, x_rationality)
        score_N = model.naturalness_score(score_T, score_R)

    print(f"The technical score of the test image is {score_T.item():.4f}")
    print(f"The rationality score of the test image is {score_R.item():.4f}")
    print(f"The naturalness score of the test image is {score_N.item():.4f}")


if __name__ == "__main__":
    main()
