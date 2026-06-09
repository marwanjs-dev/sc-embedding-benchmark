from pathlib import Path
import anndata as ad


def load_h5ad(path: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return ad.read_h5ad(str(p))
