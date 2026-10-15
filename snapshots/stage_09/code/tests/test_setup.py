"""Setup-session checks: project location, paths.yaml, no hard-coded paths, venv, CUDA, CLAUDE.md, raw data."""
import importlib
import os
import re
import subprocess
import sys

import pytest
import torch
import yaml

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, "scripts"))
PATHS_YAML = os.path.join(PROJECT_DIR, "configs", "paths.yaml")
CLAUDE_SECTIONS = ["Project goal", "Paths and environment", "Golden rules", "Architectures",
                   "Datasets", "Stage plan", "Results layout", "Commands", "Session log"]


def _paths():
    with open(PATHS_YAML) as fh:
        return yaml.safe_load(fh)


def test_project_dir_and_symlink():
    p = _paths()
    assert os.path.isdir(p["project_dir"])
    assert os.path.realpath(PROJECT_DIR) == os.path.realpath(p["project_dir"])
    assert os.path.islink(p["legacy_symlink"])
    assert os.path.realpath(p["legacy_symlink"]) == os.path.realpath(p["project_dir"])
    assert os.path.isfile(os.path.join(p["legacy_symlink"], "configs", "paths.yaml"))


def test_paths_yaml_every_path_exists():
    p = _paths()
    for key in ["project_dir", "datasets_dir", "legacy_symlink"]:
        assert os.path.exists(p[key]), key
    assert len(p["datasets"]) == 6
    for did, folder in p["datasets"].items():
        assert os.path.isdir(os.path.join(p["datasets_dir"], folder)), did


def test_datasets_dir_outside_project():
    p = _paths()
    real_ds = os.path.realpath(p["datasets_dir"])
    assert not real_ds.startswith(os.path.realpath(p["project_dir"]) + os.sep)


# Built from pieces so this file does not match its own pattern.
ABS_PATH = re.compile(r"(?<![\w.:])/(?:" + "|".join(["ho" + "me", "us" + "r", "mn" + "t", "med" + "ia", "op" + "t",
                                                      "tm" + "p", "ro" + "ot", "va" + "r", "Us" + "ers"]) + r")/"
                      + r"|~" + "/" + r"|AI_" + "project")


def _code_files():
    for sub in ["scripts", "src", "tests", "configs"]:
        for dp, dns, fns in os.walk(os.path.join(PROJECT_DIR, sub)):
            dns[:] = [d for d in dns if d != "__pycache__"]
            for fn in fns:
                path = os.path.join(dp, fn)
                if os.path.samefile(path, PATHS_YAML) or not fn.endswith((".py", ".sh", ".yaml", ".yml", ".txt")):
                    continue
                yield path


def test_no_absolute_paths_outside_paths_yaml():
    hits = []
    for path in _code_files():
        with open(path) as fh:
            for n, line in enumerate(fh, 1):
                if n == 1 and line.startswith("#!"):
                    continue  # shebang
                if ABS_PATH.search(line):
                    hits.append(f"{os.path.relpath(path, PROJECT_DIR)}:{n}: {line.strip()}")
    assert not hits, "absolute paths found:\n" + "\n".join(hits)


def test_venv_imports():
    py = os.path.join(PROJECT_DIR, ".venv", "bin", "python")
    r = subprocess.run([py, "-c", "import torch, timm, sklearn, pandas, imagehash, yaml; print('ok')"],
                       capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.strip() == "ok", r.stderr
    assert os.path.realpath(sys.prefix) == os.path.realpath(os.path.join(PROJECT_DIR, ".venv"))


def test_cuda_available():
    assert torch.cuda.is_available()


def test_check_gpu_script_exits_zero():
    r = subprocess.run([sys.executable, os.path.join(PROJECT_DIR, "scripts", "check_gpu.py")],
                       capture_output=True, text=True)
    assert r.returncode == 0 and "GPU CHECK: OK" in r.stdout, r.stdout + r.stderr


def test_claude_md_sections():
    with open(os.path.join(PROJECT_DIR, "CLAUDE.md")) as fh:
        txt = fh.read()
    for n, title in enumerate(CLAUDE_SECTIONS, 1):
        assert re.search(rf"^## {n}\. {re.escape(title)}", txt, re.M), f"missing section {n}. {title}"
    assert len(txt.splitlines()) <= 260


def _read_tsv(path):
    with open(path) as fh:
        next(fh)
        return [tuple(line.rstrip("\n").split("\t")) for line in fh]


def test_raw_data_snapshot_matches():
    snap = importlib.import_module("00_snapshot_raw")
    start = _read_tsv(os.path.join(PROJECT_DIR, "data_registry", "raw_snapshot_start.tsv"))
    live = [tuple(map(str, r)) for r in snap.snapshot()]
    assert len(start) > 0
    assert start == live, "raw data changed since the Stage 1 START snapshot"


@pytest.mark.parametrize("pat", [".venv/", "cache/", "*.pt", "logs/*.log"])
def test_gitignore(pat):
    with open(os.path.join(PROJECT_DIR, ".gitignore")) as fh:
        assert pat in fh.read().split("\n")
