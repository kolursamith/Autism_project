"""Single source of project and dataset paths: configs/paths.yaml."""
import os

import yaml

_CONFIG = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                       "configs", "paths.yaml")


def load_paths():
    with open(_CONFIG) as fh:
        return yaml.safe_load(fh)


_P = load_paths()
PROJECT_DIR = _P["project_dir"]
DATASETS_DIR = _P["datasets_dir"]
DATASETS = _P["datasets"]  # dataset_id -> folder name


def dataset_dir(dataset_id):
    return os.path.join(DATASETS_DIR, DATASETS[dataset_id])
