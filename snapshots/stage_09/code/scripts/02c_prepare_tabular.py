"""Stage 2 - tabular datasets D04, D05, D06: deterministic cleaning, column roles, duplicate groups (E).
Outputs -> cache/<dataset_id>/{clean.csv, columns.yaml, changes.csv, checks.json}. Log: logs/stage02_tabular.log
Usage: .venv/bin/python scripts/02c_prepare_tabular.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data import tabular as tb  # noqa: E402
from src.data.common import (cache_dir, get_logger, load_cfg, md_table, set_seed, write_csv, write_json,  # noqa: E402
                             write_yaml)


def main():
    set_seed(42)
    log = get_logger("stage02_tabular")
    for short in ["D04", "D05", "D06"]:
        cfg = load_cfg(short)
        out = cache_dir(cfg["dataset_id"])
        df, changes, spec, checks = tb.clean(cfg)
        log.info(f"===== {cfg['dataset_id']} ({cfg['file']}): {checks['n_rows_raw']} raw rows -> "
                 f"{checks['n_rows_clean']} clean rows; labels {checks['label_counts']}")
        log.info("changes:\n" + md_table(changes))
        log.info(f"duplicates: full rows (id ignored) = {checks['duplicate_rows_full_excluding_id']}; rows in "
                 f"identical-feature groups = {checks['rows_in_duplicate_feature_groups']} in "
                 f"{checks['duplicate_feature_groups']} groups ({checks['duplicate_groups_with_mixed_labels']} with "
                 f"mixed labels); all members of a group share group_id -> same fold. n_groups = {checks['n_groups']}")
        log.info(f"missing values per feature column: {checks['missing_per_column']}")

        if "result_column_test" in cfg:
            rc = cfg["result_column_test"]
            res = tb.result_function_test(df, rc["items"], rc["column"])
            res["decision"] = ("excluded from features (direct function of A1-A10)" if res["is_function_of_items"]
                               else "kept as a feature (NOT a direct function of A1-A10)")
            write_json(res, os.path.join(out, "result_column_test.json"))
            log.info(f"'{rc['column']}' function-of-items test: {res}")
            if res["is_function_of_items"] and rc["column"] in spec["feature_sets"]["set_all"]:
                raise SystemExit("result is a function of the items: move it to 'excluded' in the config")
        if "label_rule_check" in cfg:
            lr = cfg["label_rule_check"]
            acc, cm = tb.item_sum_rule(df, lr["items"], lr["threshold"])
            checks["label_rule_check"] = {"rule": f"label == (sum(A1..A10) >= {lr['threshold']})", "accuracy": acc,
                                          "confusion_matrix[[TN,FP],[FN,TP]]": cm}
            log.info(f"label rule re-check on clean data: {checks['label_rule_check']}")
        for name, fs in spec["feature_sets"].items():
            log.info(f"feature set {name} ({len(fs)}): {fs}")

        write_csv(df, os.path.join(out, "clean.csv"))
        write_csv(changes, os.path.join(out, "changes.csv"))
        write_yaml(spec, os.path.join(out, "columns.yaml"))
        write_json(checks, os.path.join(out, "checks.json"))


if __name__ == "__main__":
    main()
