"""D02 facial images (DFDK folder): pooled manifest + pHash near-duplicate groups (DD-05, DD-07).
Only the D02 folder is read; D03 (AutismDataset, a copy of D02) is never opened (DD-06)."""
import hashlib
import os
import re

import imagehash
import numpy as np
import pandas as pd
from PIL import Image
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from src.data.common import DATASETS_DIR, dataset_dir

IMG_EXT = {".jpg", ".jpeg", ".png"}
_POP8 = np.array([bin(i).count("1") for i in range(256)], np.uint8)


def norm_class(name):
    return re.sub(r"[^a-z]", "", name.lower())  # 'Non-Autistic' / 'Non_Autistic' -> 'nonautistic'


def list_images(cfg):
    """Images inside <split>/<class>/ folders; returns DataFrame + the raw class-folder names per split."""
    root = dataset_dir("D02_facial_images_A")
    rows, raw_names = [], {}
    for dp, dns, fns in os.walk(root):
        dns.sort()
        parts = os.path.relpath(dp, root).split(os.sep)
        if len(parts) < 2 or parts[-2] not in cfg["splits"]:
            continue
        split, cls = parts[-2], parts[-1]
        raw_names.setdefault(split, set()).add(cls)
        nc = norm_class(cls)
        if nc not in cfg["class_map"]:
            raise ValueError(f"unknown class folder {cls!r}")
        for fn in sorted(fns):
            if os.path.splitext(fn)[1].lower() not in IMG_EXT:
                continue
            p = os.path.join(dp, fn)
            rows.append({"image_id": f"{split}/{nc}/{fn}", "rel_path": os.path.relpath(p, DATASETS_DIR),
                         "label": cfg["class_map"][nc], "original_split": split})
    df = pd.DataFrame(rows).sort_values("image_id").reset_index(drop=True)
    return df, {k: sorted(v) for k, v in sorted(raw_names.items())}


def describe(df):
    """md5, 64-bit pHash (hex), width, height, mode for every image."""
    out = []
    for p in df["rel_path"]:
        full = os.path.join(DATASETS_DIR, p)
        with open(full, "rb") as fh:
            md5 = hashlib.md5(fh.read()).hexdigest()
        with Image.open(full) as im:
            out.append({"md5": md5, "phash": str(imagehash.phash(im)), "width": im.width, "height": im.height,
                        "mode": im.mode})
    return pd.concat([df, pd.DataFrame(out, index=df.index)], axis=1)


def hamming_matrix(phash_hex):
    v = np.array([int(h, 16) for h in phash_hex], dtype=np.uint64)
    x = (v[:, None] ^ v[None, :]).view(np.uint8).reshape(len(v), len(v), 8)
    return _POP8[x].sum(axis=2, dtype=np.int32)


def near_duplicate_groups(df, max_dist):
    """Edges = pairs with Hamming <= max_dist; group = connected component. group_id numbered in image_id order."""
    d = hamming_matrix(df["phash"].tolist())
    i, j = np.nonzero(np.triu(d <= max_dist, k=1))
    pairs = pd.DataFrame({"image_a": df["image_id"].to_numpy()[i], "image_b": df["image_id"].to_numpy()[j],
                          "hamming": d[i, j]})
    n = len(df)
    g = coo_matrix((np.ones(len(i)), (i, j)), shape=(n, n))
    _, comp = connected_components(g, directed=False)
    # renumber components by first appearance (df is sorted by image_id) -> deterministic ids
    order = {}
    for c in comp:
        order.setdefault(c, len(order))
    gid = np.array([order[c] for c in comp])
    return [f"g{k:04d}" for k in gid], pairs
