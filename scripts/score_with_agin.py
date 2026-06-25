
import argparse
import subprocess
from pathlib import Path


def parse_scores(stdout: str):
    scores = {
        "technical": None,
        "rationality": None,
        "naturalness": None,
    }

    for line in stdout.splitlines():
        lower = line.lower()
        for key in scores:
            if f"{key} score" in lower:
                scores[key] = float(line.split()[-1])

    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    args = parser.parse_args()

    agin_joint = args.project / "external" / "AGIN" / "JOINT"
    model_path = args.project / "checkpoints" / "JOINT_2024.pth"

    if not agin_joint.exists():
        raise FileNotFoundError(f"Missing AGIN JOINT folder: {agin_joint}")

    if not model_path.exists():
        raise FileNotFoundError(f"Missing JOINT checkpoint: {model_path}")

    if not args.image.exists():
        raise FileNotFoundError(f"Missing image: {args.image}")

    cmd = [
        "python",
        "test_single_image.py",
        "--model_path", str(model_path),
        "--model", "JOINT",
        "--image_path", str(args.image),
        "--resize", "384",
        "--crop_size", "224",
        "--gpu_ids", "0",
    ]

    result = subprocess.run(
        cmd,
        cwd=agin_joint,
        capture_output=True,
        text=True,
    )

    print("STDOUT:")
    print(result.stdout)

    if result.stderr.strip():
        print("STDERR:")
        print(result.stderr)

    result.check_returncode()

    scores = parse_scores(result.stdout)
    print("PARSED_SCORES:", scores)


if __name__ == "__main__":
    main()
