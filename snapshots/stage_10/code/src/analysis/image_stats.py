"""Simple NON-facial image statistics for the D02 dataset-artefact baseline (Stage 8 C2). Raw images are only read."""
import os

import cv2
import numpy as np
from PIL import Image

# libjpeg standard luminance quantisation table (ITU T.81 Annex K), quality 50
STD_LUMA = np.array([16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56,
                     14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
                     49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99], dtype=float)
FEATURES = ["width", "height", "aspect_ratio", "file_size_kb", "jpeg_quality", "mean_r", "mean_g", "mean_b", "std_r",
            "std_g", "std_b", "brightness_mean", "brightness_std", "border_r", "border_g", "border_b", "sharpness_log_lapvar"]


def jpeg_quality(im):
    """IJG quality estimate from the luminance table (order-invariant: ratio of table sums); NaN if not a JPEG."""
    q = getattr(im, "quantization", None)
    if not q or 0 not in q:
        return float("nan")
    scale = 100.0 * float(np.sum(q[0])) / STD_LUMA.sum()
    qual = (200.0 - scale) / 2.0 if scale <= 100 else 5000.0 / scale
    return float(np.clip(qual, 1, 100))


def image_features(path, border=0.10):
    with Image.open(path) as im:
        jq = jpeg_quality(im)
        rgb = np.asarray(im.convert("RGB"), dtype=np.float32)
    h, w = rgb.shape[:2]
    gray = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    bh, bw = max(1, int(round(h * border))), max(1, int(round(w * border)))
    mask = np.zeros((h, w), bool)
    mask[:bh], mask[-bh:], mask[:, :bw], mask[:, -bw:] = True, True, True, True
    bord = rgb[mask].mean(axis=0)
    lap = cv2.Laplacian(gray, cv2.CV_64F).var()
    return {"width": w, "height": h, "aspect_ratio": w / h, "file_size_kb": os.path.getsize(path) / 1024.0,
            "jpeg_quality": jq, "mean_r": float(rgb[..., 0].mean()), "mean_g": float(rgb[..., 1].mean()),
            "mean_b": float(rgb[..., 2].mean()), "std_r": float(rgb[..., 0].std()), "std_g": float(rgb[..., 1].std()),
            "std_b": float(rgb[..., 2].std()), "brightness_mean": float(gray.mean()), "brightness_std": float(gray.std()),
            "border_r": float(bord[0]), "border_g": float(bord[1]), "border_b": float(bord[2]),
            "sharpness_log_lapvar": float(np.log1p(lap))}
