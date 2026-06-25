
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def import_agin_minimizer(project: Path):
    utils_path = project / "external" / "AGIN" / "JOINT" / "utils"
    sys.path.insert(0, str(utils_path))
    from AmbrosioTortorelliMinimizer import AmbrosioTortorelliMinimizer
    return AmbrosioTortorelliMinimizer


def compute_lfm(image: Image.Image, project: Path):
    """
    Compute the low-frequency map used by JOINT's rationality branch.
    """
    minimizer_cls = import_agin_minimizer(project)

    result = []
    img = np.array(image)

    for channel in cv2.split(img):
        solver = minimizer_cls(
            channel,
            iterations=1,
            tol=0.1,
            solver_maxiterations=6,
        )
        f, _ = solver.minimize()
        result.append(f)

    f = cv2.merge(result)
    cv2.normalize(f, f, 0, 255, cv2.NORM_MINMAX)
    return np.float32(f)


def compute_lfm_pil(image: Image.Image, project: Path):
    lfm = compute_lfm(image, project)
    return Image.fromarray(np.uint8(lfm))
