
from pathlib import Path
import random

import cv2
import numpy as np
import torch
from PIL import Image


def numpy_to_tensor(img: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()


def prepare_pal_input(img: np.ndarray, device: torch.device) -> torch.Tensor:
    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([58.395, 57.12, 57.375], dtype=np.float32).reshape(1, 1, 3)
    normalized = (img.astype(np.float32) - mean) / std
    return numpy_to_tensor(normalized).to(device)


def load_pal_model(checkpoint_path: Path, device: torch.device):
    """
    Load PAL artifact/localization model used before JOINT technical scoring.
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing PAL checkpoint: {checkpoint_path}")

    try:
        model = torch.jit.load(str(checkpoint_path), map_location=device)
    except Exception:
        model = torch.load(str(checkpoint_path), map_location=device, weights_only=False)

    return model.to(device).eval()


def patch_has_artifact(mask: np.ndarray, row: int, col: int, patch_size: int) -> bool:
    row0 = row * patch_size
    row1 = (row + 1) * patch_size
    col0 = col * patch_size
    col1 = (col + 1) * patch_size
    return bool(np.any(mask[row0:row1, col0:col1] != 0))


def patch_partition_image(
    image_path: Path,
    output_path: Path,
    pal_model,
    device: torch.device,
    size: int = 224,
    patch_size: int = 32,
    seed: int = 1234,
):
    """
    Create the patch-partitioned image used by JOINT's technical branch.
    """
    random.seed(seed)

    image_path = Path(image_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    img_pil = Image.open(image_path).convert("RGB")
    img_resized = np.array(img_pil.resize((size, size)).convert("RGB"))

    pal_input = prepare_pal_input(img_resized, device)

    with torch.no_grad():
        pal = pal_model(pal_input).detach().cpu().numpy()[0][0]

    artifact_mask = pal != 0

    resized_image = cv2.resize(np.array(img_pil), (size, size))

    num_rows = resized_image.shape[0] // patch_size
    num_cols = resized_image.shape[1] // patch_size

    patches = np.zeros(
        (num_rows, num_cols, patch_size, patch_size, 3),
        dtype=np.uint8,
    )
    swapped = np.zeros((num_rows, num_cols), dtype=bool)

    for row in range(num_rows):
        for col in range(num_cols):
            patches[row, col] = resized_image[
                row * patch_size:(row + 1) * patch_size,
                col * patch_size:(col + 1) * patch_size,
            ]

    for row in range(num_rows):
        for col in range(num_cols):
            if swapped[row, col]:
                continue

            if patch_has_artifact(artifact_mask, row, col, patch_size):
                continue

            neighbors = [
                (row - 1, col - 1), (row - 1, col), (row - 1, col + 1),
                (row, col - 1),                     (row, col + 1),
                (row + 1, col - 1), (row + 1, col), (row + 1, col + 1),
            ]

            valid_neighbors = [
                (nr, nc)
                for nr, nc in neighbors
                if 0 <= nr < num_rows
                and 0 <= nc < num_cols
                and not swapped[nr, nc]
                and not patch_has_artifact(artifact_mask, nr, nc, patch_size)
            ]

            if not valid_neighbors:
                continue

            nr, nc = random.choice(valid_neighbors)

            patches[row, col], patches[nr, nc] = (
                patches[nr, nc].copy(),
                patches[row, col].copy(),
            )

            swapped[row, col] = True
            swapped[nr, nc] = True

    shuffled_image = np.zeros_like(resized_image)

    for row in range(num_rows):
        for col in range(num_cols):
            shuffled_image[
                row * patch_size:(row + 1) * patch_size,
                col * patch_size:(col + 1) * patch_size,
            ] = patches[row, col]

    cv2.imwrite(str(output_path), cv2.cvtColor(shuffled_image, cv2.COLOR_RGB2BGR))

    return output_path
