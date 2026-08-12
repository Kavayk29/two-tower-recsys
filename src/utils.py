"""
Place at: src/utils.py

Single source of truth for where model artifacts (checkpoints, FAISS
index) live. Both trainer.py and build_index.py previously computed this
path independently and disagreed (one used "artifacts_dir", the other
"artifacts" -- and the Kaggle branches didn't match either), so
build_index.py silently couldn't find what trainer.py had just saved.
"""

import os
from pathlib import Path


def get_artifacts_dir(config: dict | None = None) -> Path:
    # 1. Explicit override in config.yaml (data.artifacts_dir), if present
    if config and config.get("data", {}).get("artifacts_dir"):
        path = Path(config["data"]["artifacts_dir"])
    # 2. Kaggle environment
    elif os.path.exists("/kaggle/working"):
        path = Path("/kaggle/working/artifacts")
    # 3. Local / any other environment
    else:
        path = Path("artifacts")

    path.mkdir(parents=True, exist_ok=True)
    return path