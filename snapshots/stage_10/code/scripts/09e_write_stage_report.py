"""Stage 9 E - reports/STAGE_09_REPORT.md (stage record per CLAUDE.md): pre-flight, tables, figures, final report,
read-only check of results/, pytest. Usage: .venv/bin/python scripts/09e_write_stage_report.py"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd  # noqa: E402

from src.analysis.final import FIG_FINAL, FINAL_DIR, results_manifest_lines  # noqa: E402
from src.data.common import LOG_DIR, PROJECT_DIR, md_table  # noqa: E402

TABLES = [("master_table", "one row per dataset x model (50): fold mean ± std of 7 metrics, pooled AUC + 95% CI, ECE; `__source` per number"),
          ("shortcut_table", "D01 model vs tracking-quality baseline (child AUC, difference + CI, DeLong, matched AUC + CI, Spearman, rule) and D02 vs image-statistics baseline, near-duplicate removal, original split (long format, source per value)"),
          ("stats_table", "Friedman / Iman-Davenport / Nemenyi / average ranks / Nadeau-Bengio + Holm per analysis; pretrained vs scratch (long format)"),
          ("fold_auc_matrix", "per-fold test ROC-AUC for every dataset x model (appendix)"),
          ("datasets_table", "dataset facts (counts from the frozen split files), roles, exclusions"),
          ("reproducibility_table", "environment, GPU/CPU time, snapshot code hashes"),
          ("confusion_matrices_best", "pooled OOF confusion matrices of the best model per dataset (F8 data)"),
          ("reliability_bins_best", "reliability bins of the best model per dataset (F12 data)"),
          ("d01_m07_attention_per_second", "M07 mean attention per second, ASD vs TD, per-child averaged (F9 data)")]
FIGS = [("F01_pipeline", "pipeline diagram: D01 / D02 inputs -> preprocessing -> spatial encoder -> projection -> temporal encoder -> attention -> P(ASD), evaluation band"),
        ("F02_d01_auc_vs_quality", "D01 pooled AUC + 95% CI per model, clip and child panels, quality-only reference lines (clip 0.678; child LR 0.823 with CI band)"),
        ("F03_d01_forest_vs_quality", "D01 forest plot of child AUC difference model - quality LR; all CIs cross or lie below 0, M07 the only point right of 0"),
        ("F04_d01_matched_subset", "D01 AUC on 14 quality-matched pairs with CIs; only M07's CI clear of 0.5; quality-only band 0.45-0.68"),
        ("F05_d02_auc_vs_image_stats", "D02 pooled AUC + CI per model (0.867-0.898) far right of the image-statistics baselines (0.781 / 0.797, CI bands)"),
        ("F06_d02_roc_top3", "D02 pooled ROC curves of M07, M04, M06 and the image-statistics random forest"),
        ("F07_cd_diagrams", "CD diagrams: D01 clip one bar joins all 10 models; D02 M04 / M07 best ranks, custom CNNs worst, 4 overlapping bars"),
        ("F08_confusion_matrices", "confusion matrices (counts + row %) of the best model per dataset (7 panels + note)"),
        ("F09_d01_m07_attention", "M07 attention per second: near uniform for seconds 1-6, rising at second 8, higher for ASD (0.172) than TD (0.141)"),
        ("F10_tabular_auc", "tabular pooled AUC + CI per model for D04, D05 set_all, D05 set_honest, D06 with the 0.5 dummy line"),
        ("F11_pretrained_vs_scratch", "per-model fold-mean AUC by encoder group with group means (D01 clip / child, D02); pretrained group higher in all three"),
        ("F12_reliability_best", "reliability diagrams of the best model per dataset with ECE in the titles")]


def main():
    pre = json.load(open(os.path.join(FINAL_DIR, "preflight.json")))
    rec = open(os.path.join(FINAL_DIR, "preflight_results_manifest.sha256")).read().splitlines()
    cur = results_manifest_lines()
    same = rec == cur
    with open(os.path.join(LOG_DIR, "stage09_results_unchanged.txt"), "w") as fh:
        fh.write(f"results/ except results/final/: {len(rec)} files at pre-flight, {len(cur)} now; identical: {same}\n")
    pyt = open(os.path.join(LOG_DIR, "stage09_pytest.txt")).read().rstrip("\n")
    last = pyt.splitlines()[-1]
    py_ok = re.search(r"(\d+) (failed|error)", last) is None and "passed" in last
    tab_rows = []
    for t, d in TABLES:
        df = pd.read_csv(os.path.join(FINAL_DIR, f"{t}.csv"))
        tab_rows.append({"file": f"results/final/{t}.csv" + (" + .md" if os.path.isfile(os.path.join(FINAL_DIR, f"{t}.md")) else ""),
                         "rows": len(df), "content": d})
    figs_ok = all(os.path.getsize(os.path.join(FIG_FINAL, f"{f}.{e}")) > 0 for f, _ in FIGS for e in ["png", "svg"])
    md = os.path.join(PROJECT_DIR, "reports", "FINAL_RESULTS_REPORT.md")
    html = os.path.join(PROJECT_DIR, "reports", "FINAL_RESULTS_REPORT.html")
    rep_ok = os.path.isfile(md) and os.path.isfile(html)
    abstract = open(md).read().split("## 1. Abstract", 1)[1].split("## 2.", 1)[0].strip()
    L = ["# STAGE 9 REPORT - Final figures, tables and results report", "",
         "Screening-support research, not diagnosis. Stage 9 collects and presents; it trains nothing and recomputes no "
         "metric (results/D0*/ and results/stage8/ read-only; byte-identical before/after, section 5). Code: "
         "`src/analysis/final.py`, `scripts/09a_preflight.py`, `09b_tables.py`, `09c_figures.py`, `09d_write_report.py`, "
         "`09e_write_stage_report.py`; tests `tests/test_stage09.py`. CPU only.", "",
         "## 1. Pre-flight (`logs/stage09_preflight.txt`)", "", "```", open(os.path.join(LOG_DIR, "stage09_preflight.txt")).read().rstrip(), "```", "",
         "## 2. Tables (`results/final/`)", "", md_table(pd.DataFrame(tab_rows)), "",
         "Traceability: every number in master_table.csv has a `<column>__source` reference (`file.json::key::...` or "
         "`file.csv::col=val&...::column`); the long tables carry a `source` column. `tests/test_stage09.py` re-reads every "
         "referenced value (master: 50 x 19 numbers; shortcut, stats and fold tables: all values) and requires equality.", "",
         "## 3. Figures (`reports/figures/final/`, 300 dpi PNG + SVG; one colour per model in all figures)", "",
         "Each figure was opened and checked for overlapping labels; fixes applied: F2 x-labels shortened (they collided), F12 "
         "x tick labels added on the D04 panel, F1 annotation moved, F7 height reduced.", "",
         md_table(pd.DataFrame([{"figure": f, "content": d} for f, d in FIGS])), "",
         "## 4. Final results report", "",
         f"`reports/FINAL_RESULTS_REPORT.md` (11 sections + figures; abstract {len(abstract.split())} words) and "
         "`reports/FINAL_RESULTS_REPORT.html` (pandoc 3.1.3, `--standalone --embed-resources`, 12 figures embedded as base64; "
         "python-markdown is not installed, pandoc is). All result numbers in it come from `results/final/*.csv`; the Stage 8 "
         "interpretation rule and conclusions are copied verbatim from `reports/STAGE_08_REPORT.md`.", "",
         "Abstract:", "", "> " + abstract, "",
         "## 5. Read-only check", "", "```", open(os.path.join(LOG_DIR, "stage09_results_unchanged.txt")).read().rstrip(), "```", "",
         "## 6. Tests (`.venv/bin/python -m pytest -v`, `logs/stage09_pytest.txt`)", "",
         f"Result: {last.strip('= ')}", "", "<details><summary>full pytest output</summary>", "", "```", pyt, "```", "", "</details>", ""]
    ok = pre["preflight_ok"] and same and py_ok and figs_ok and rep_ok
    why = [n for n, v in [("pre-flight", pre["preflight_ok"]), ("results unchanged", same), ("pytest", py_ok), ("figures", figs_ok),
                          ("final report", rep_ok)] if not v]
    L.append("STAGE 9 DECISION: PASS" if ok else "STAGE 9 DECISION: FAIL - " + ", ".join(why))
    with open(os.path.join(PROJECT_DIR, "reports", "STAGE_09_REPORT.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")
    print(L[-1])


if __name__ == "__main__":
    main()
