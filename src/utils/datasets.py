"""Dataset registry and downloader.

Usage examples:
  python src/utils/datasets.py list
  python src/utils/datasets.py download pbmc3k pbmc4k --out data/raw

The script will attempt to use `scanpy.datasets` or `scvi.data` when configured,
or download a direct `url` if provided in `configs/datasets.yaml`.
"""
import argparse
import yaml
from pathlib import Path
import importlib
import requests
import sys


def load_registry():
    p = Path("configs/datasets.yaml")
    if not p.exists():
        raise FileNotFoundError("configs/datasets.yaml not found")
    return yaml.safe_load(open(p))


def save_adata(adata, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(str(out_path))


def download_url(url: str, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} -> {out_path}")
    r = requests.get(url, stream=True)
    r.raise_for_status()
    with open(out_path, "wb") as fh:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                fh.write(chunk)
    return out_path


def fetch(name: str, entry: dict, out_dir: Path):
    provider = entry.get("provider")
    out_path = out_dir / f"{name}.h5ad"

    if provider == "scanpy":
        fn = entry.get("fn")
        sd = importlib.import_module("scanpy.datasets")
        if hasattr(sd, fn):
            print(f"Fetching {name} using scanpy.datasets.{fn}()")
            ad = getattr(sd, fn)()
            save_adata(ad, out_path)
            return out_path
        else:
            raise RuntimeError(f"scanpy.datasets does not have function {fn}")

    if provider == "scvi":
        fn = entry.get("fn")
        try:
            sd = importlib.import_module("scvi.data")
            if hasattr(sd, fn):
                print(f"Fetching {name} using scvi.data.{fn}()")
                ad = getattr(sd, fn)()
                save_adata(ad, out_path)
                return out_path
            else:
                raise RuntimeError(f"scvi.data does not have function {fn}")
        except ModuleNotFoundError:
            raise RuntimeError("scvi is not installed in the active environment")

    if provider == "url":
        url = entry.get("url")
        if not url:
            raise RuntimeError(f"No url provided for dataset {name}")
        return download_url(url, out_path)

    raise RuntimeError(f"Unknown provider {provider} for dataset {name}")


def main():
    parser = argparse.ArgumentParser(description="Dataset registry and downloader")
    sub = parser.add_subparsers(dest="cmd")
    sub.add_parser("list")
    dwn = sub.add_parser("download")
    dwn.add_argument("names", nargs="+")
    dwn.add_argument("--out", default="data/raw")

    args = parser.parse_args()
    reg = load_registry().get("datasets", {})

    if args.cmd == "list":
        for k, v in reg.items():
            print(f"{k}: provider={v.get('provider')} note={v.get('note','')}")
        return

    if args.cmd == "download":
        out_dir = Path(args.out)
        for name in args.names:
            if name not in reg:
                print(f"Unknown dataset: {name}")
                continue
            entry = reg[name]
            try:
                path = fetch(name, entry, out_dir)
                print(f"Saved dataset to {path}")
            except Exception as e:
                print(f"Failed to fetch {name}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
