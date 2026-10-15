"""Stage 2 - D02 facial images: pooled manifest + near-duplicate groups (D1-D3). Outputs -> cache/D02_facial_images_A/.
Re-runnable: overwrites its outputs. Log: logs/stage02_d02.log
Usage: .venv/bin/python scripts/02b_prepare_d02.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data import d02_faces as f  # noqa: E402
from src.data.common import cache_dir, get_logger, load_cfg, md_table, set_seed, write_csv, write_json  # noqa: E402

DS = "D02_facial_images_A"


def main():
    cfg = load_cfg("D02")
    set_seed(cfg["seed"])
    log = get_logger("stage02_d02")
    out = cache_dir(DS)

    df, raw_names = f.list_images(cfg)
    log.info(f"D1: real class folder names per split: {raw_names}")
    log.info(f"D1: pooled {len(df)} images; per split x label:\n"
             + md_table(df.groupby(["original_split", "label"]).size().rename("n").reset_index()))
    df = f.describe(df)
    log.info(f"D2: modes {df['mode'].value_counts().to_dict()}; width {df.width.min()}-{df.width.max()}, "
             f"height {df.height.min()}-{df.height.max()}; md5 duplicates {int(df.md5.duplicated().sum())}")

    mx = cfg["near_duplicate"]["max_hamming"]
    df["group_id"], pairs = f.near_duplicate_groups(df, mx)
    write_csv(pairs, os.path.join(out, "near_duplicate_pairs.csv"))
    sizes = df.groupby("group_id").size()
    log.info(f"D2: {len(pairs)} near-duplicate pairs (Hamming <= {mx}); {sizes.size} groups; group size distribution "
             f"{sizes.value_counts().sort_index().to_dict()}")
    lab = df.groupby("group_id")["label"].nunique()
    mixed = sorted(lab[lab > 1].index)
    conflicts = df[df.group_id.isin(mixed)]
    write_csv(conflicts, os.path.join(out, "label_conflicts_excluded.csv"))
    log.info(f"D2: groups with both labels (label conflicts): {len(mixed)} {mixed}; images excluded: {len(conflicts)}")
    multi = df[df.group_id.isin(sizes[sizes > 1].index)]
    log.info("D2: near-duplicate groups (size > 1):\n"
             + md_table(multi[["group_id", "image_id", "label", "original_split"]]))
    log.info("D2: groups spanning original splits: "
             f"{int((multi.groupby('group_id').original_split.nunique() > 1).sum())}")

    if cfg["near_duplicate"]["drop_mixed_label_groups"]:
        df = df[~df.group_id.isin(mixed)]
    cols = ["image_id", "rel_path", "label", "original_split", "md5", "phash", "group_id", "width", "height", "mode"]
    write_csv(df[cols], os.path.join(out, "manifest.csv"))
    summ = {"n_images": len(df), "label_counts": df.label.value_counts().to_dict(), "n_groups": int(df.group_id.nunique()),
            "n_pairs": len(pairs), "group_size_distribution": sizes.value_counts().sort_index().to_dict(),
            "label_conflict_groups": mixed, "n_excluded_conflict_images": len(conflicts),
            "raw_class_folder_names": raw_names}
    write_json(summ, os.path.join(out, "d02_summary.json"))
    log.info(f"D3: manifest.csv written: {len(df)} images, labels {summ['label_counts']} (1=autistic)")


if __name__ == "__main__":
    main()
