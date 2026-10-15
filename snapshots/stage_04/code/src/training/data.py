"""Datasets for training, built ONLY from the sample ids of splits/<ds>/fold_XX.json (data contracts, CLAUDE.md §5).

Every item is (x, y, index); `index` is the position in the dataset's own id list, so the engine can log exactly which
sample ids went into which batch (test-id leakage assert).
  D01FeatureDataset  frozen backbones: cached per-frame features (8, C) float16 -> float32
  D01ClipDataset     trainable (custom CNN) backbones: uint8 clips (8, 3, 224, 224) from clips_uint8.npy (memmap)
  D02ImageDataset    uint8 image (3, 224, 224): train = RandomResizedCrop/flip/ColorJitter, eval = resize + center crop
Normalisation happens inside the model (model contracts), so loaders return raw uint8 / cached features.
Data loaders are re-created every epoch with a torch.Generator seeded by (seed, epoch, split): the shuffle order and the
worker seeds (augmentation) depend only on (seed, epoch), which makes resume-from-checkpoint exact."""
import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

from src.data.common import CACHE_DIR, DATASETS_DIR, SPLITS_DIR
from src.models.feature_cache import CLIPS_NPY, load_cache

D01, D02 = "D01_eye_tracking", "D02_facial_images_A"
ALIASES = {"D01": D01, "D02": D02, D01: D01, D02: D02}


def sample_table(ds, splits_root=SPLITS_DIR):
    """sample_id, group_id, label (from the hash-verified splits/<ds>/folds.csv) + where the data lives."""
    t = pd.read_csv(os.path.join(splits_root, ds, "folds.csv"))
    if ds == D01:
        idx = pd.read_csv(os.path.join(CACHE_DIR, D01, "clips_index.csv"))
        src = idx.rename(columns={"clip_id": "sample_id"})[["sample_id", "row_in_npy", "label"]]
    elif ds == D02:
        man = pd.read_csv(os.path.join(CACHE_DIR, D02, "manifest.csv"))
        src = man.rename(columns={"image_id": "sample_id"})[["sample_id", "rel_path", "label"]]
    else:
        raise ValueError(ds)
    m = t.merge(src, on="sample_id", how="left", suffixes=("", "_src"), validate="one_to_one")
    if m["label_src"].isna().any() or (m["label"] != m["label_src"]).any():
        raise RuntimeError(f"{ds}: folds.csv labels / ids do not match the data manifest")
    return m.drop(columns="label_src").set_index("sample_id", drop=False)


class _Base(Dataset):
    def __init__(self, table, ids):
        self.ids = list(ids)
        self.t = table.loc[self.ids]
        self.y = torch.tensor(self.t["label"].to_numpy(), dtype=torch.float32)
        self.groups = self.t["group_id"].to_numpy()

    def __len__(self):
        return len(self.ids)


class D01FeatureDataset(_Base):
    def __init__(self, table, ids, timm_name):
        super().__init__(table, ids)
        cache, self.cache_meta = load_cache(timm_name)
        self.x = torch.from_numpy(np.array(cache[self.t["row_in_npy"].to_numpy()]))  # (N, 8, C) float16, in RAM

    def __getitem__(self, i):
        return self.x[i].float(), self.y[i], i


class D01ClipDataset(_Base):
    def __init__(self, table, ids):
        super().__init__(table, ids)
        self.rows = self.t["row_in_npy"].to_numpy()
        self._clips = None  # opened lazily (per worker process)

    def __getitem__(self, i):
        if self._clips is None:
            self._clips = np.load(CLIPS_NPY, mmap_mode="r")
        return torch.from_numpy(np.array(self._clips[self.rows[i]])), self.y[i], i


def d02_transforms(cfg, train):
    if train:
        a = cfg["train_transform"]
        cj = a["color_jitter"]
        return v2.Compose([
            v2.PILToTensor(),
            v2.RandomResizedCrop(a["random_resized_crop"]["size"], scale=tuple(a["random_resized_crop"]["scale"]),
                                 antialias=True),
            v2.RandomHorizontalFlip(a["horizontal_flip_p"]),
            v2.ColorJitter(brightness=cj["brightness"], contrast=cj["contrast"], saturation=cj["saturation"])])
    e = cfg["eval_transform"]
    return v2.Compose([v2.PILToTensor(), v2.Resize(e["resize_shorter_side"], antialias=True),
                       v2.CenterCrop(e["center_crop"])])


class D02ImageDataset(_Base):
    def __init__(self, table, ids, transform):
        super().__init__(table, ids)
        self.paths = [os.path.join(DATASETS_DIR, p) for p in self.t["rel_path"]]
        self.transform = transform

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            x = self.transform(im.convert("RGB"))
        return x, self.y[i], i


def build_datasets(ds, fold_split, cfg, model, subset=None):
    """{'train','val','test'} datasets from the fold json; subset=n keeps the first n ids per class per split (tests)."""
    table = sample_table(ds)
    dcfg = cfg["data"][ds]
    parts = {}
    for split in ["train", "val", "test"]:
        ids = fold_split[split]
        if subset:
            lab = table.loc[ids, "label"]
            keep = set(lab[lab == 0].index[:subset]) | set(lab[lab == 1].index[:subset])
            ids = [i for i in ids if i in keep]
        if ds == D01 and model.spatial.policy == "frozen":
            parts[split] = D01FeatureDataset(table, ids, model.spatial.timm_name)
        elif ds == D01:
            parts[split] = D01ClipDataset(table, ids)
        else:
            parts[split] = D02ImageDataset(table, ids, d02_transforms(dcfg, train=split == "train"))
    return parts


def _worker_init(worker_id):
    s = torch.initial_seed() % 2 ** 32  # = loader base seed + worker id (drawn from the per-epoch generator)
    np.random.seed(s)
    import random
    random.seed(s)


def make_loader(dataset, batch_size, shuffle, seed, epoch, split_code, num_workers, pin_memory=True):
    g = torch.Generator()
    g.manual_seed(int(seed) * 100003 + int(epoch) * 101 + int(split_code))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=g, num_workers=num_workers,
                      pin_memory=pin_memory, worker_init_fn=_worker_init, drop_last=False,
                      persistent_workers=False)


def default_num_workers():
    return min(8, os.cpu_count() or 1)
