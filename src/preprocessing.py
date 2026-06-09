"""Preprocess raw AnnData files into benchmark-ready processed datasets.

Pipeline:
  raw AnnData
    -> standardize metadata
    -> basic QC metrics
    -> cell / gene filtering
    -> save raw counts in adata.layers['counts']
    -> normalize + log1p in adata.X
    -> highly variable gene selection
    -> subset to HVGs
    -> save processed .h5ad for PCA / scVI / scGPT

The script is intentionally conservative and config-driven so it can be reused
across PBMC and pancreas style datasets.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import yaml


@dataclass
class PreprocessConfig:
	min_genes: int = 200
	min_cells: int = 3
	max_pct_mt: float = 20.0
	min_counts: int | None = None
	n_top_genes: int = 2000
	hvg_flavor: str = "cell_ranger"
	subset_hvgs: bool = True
	make_unique: bool = True
	copy_input: bool = False


def load_config(path: str | Path) -> dict[str, Any]:
	with open(path, "r", encoding="utf-8") as handle:
		return yaml.safe_load(handle) or {}


def read_raw_adata(path: str | Path) -> ad.AnnData:
	"""Read raw single-cell input from either h5ad or legacy 10x HDF5."""
	path = Path(path)
	try:
		return sc.read_h5ad(str(path))
	except TypeError as exc:
		message = str(exc)
		if "AnnData.__init__() got an unexpected keyword argument 'matrix'" not in message:
			raise
		adata = sc.read_10x_h5(str(path))
		if "gene_ids" not in adata.var.columns:
			adata.var["gene_ids"] = adata.var_names.astype(str)
		if "gene_symbols" not in adata.var.columns:
			adata.var["gene_symbols"] = adata.var_names.astype(str)
		return adata


def _standardize_column_names(frame: pd.DataFrame, synonyms: dict[str, str]) -> pd.DataFrame:
	rename_map: dict[str, str] = {}
	lower_lookup = {column.lower(): column for column in frame.columns}
	for source, target in synonyms.items():
		if source in frame.columns:
			rename_map[source] = target
		elif source.lower() in lower_lookup:
			rename_map[lower_lookup[source.lower()]] = target
	return frame.rename(columns=rename_map)


def standardize_metadata(adata: ad.AnnData) -> ad.AnnData:
	if adata.obs_names.has_duplicates:
		adata.obs_names_make_unique()
	if adata.var_names.has_duplicates:
		adata.var_names_make_unique()

	adata.obs = _standardize_column_names(
		adata.obs,
		{
			"celltype": "cell_type",
			"cell_type": "cell_type",
			"labels": "labels",
			"batch": "batch",
			"sample": "sample",
			"condition": "condition",
			"donor": "donor",
			"patient": "patient",
		},
	)
	adata.var = _standardize_column_names(
		adata.var,
		{
			"gene_ids": "gene_ids",
			"gene_id": "gene_ids",
			"feature_types": "feature_types",
			"gene_symbols": "gene_symbols",
			"gene_symbol": "gene_symbols",
		},
	)

	if "gene_symbols" in adata.var.columns:
		adata.var_names = adata.var["gene_symbols"].astype(str)
		if adata.var_names.has_duplicates:
			adata.var_names_make_unique()

	return adata


def add_qc_flags(adata: ad.AnnData) -> ad.AnnData:
	genes = adata.var_names.astype(str)
	adata.var["mt"] = genes.str.upper().str.startswith("MT-") | genes.str.upper().str.startswith("MT.") | genes.str.upper().str.startswith("MT")
	return adata


def compute_basic_qc(adata: ad.AnnData) -> ad.AnnData:
	if "mt" not in adata.var.columns:
		adata = add_qc_flags(adata)
	sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True, percent_top=None, log1p=False)
	return adata


def filter_cells_and_genes(adata: ad.AnnData, config: PreprocessConfig) -> ad.AnnData:
	if config.min_counts is not None:
		sc.pp.filter_cells(adata, min_counts=config.min_counts)
	sc.pp.filter_cells(adata, min_genes=config.min_genes)
	sc.pp.filter_genes(adata, min_cells=config.min_cells)
	if "pct_counts_mt" in adata.obs:
		adata = adata[adata.obs["pct_counts_mt"] <= config.max_pct_mt].copy()
	return adata


def normalize_and_select_hvgs(adata: ad.AnnData, config: PreprocessConfig) -> ad.AnnData:
	adata.layers["counts"] = adata.X.copy()
	sc.pp.normalize_total(adata, target_sum=1e4)
	sc.pp.log1p(adata)
	sc.pp.highly_variable_genes(
		adata,
		layer="counts",
		flavor=config.hvg_flavor,
		n_top_genes=config.n_top_genes,
		subset=False,
		inplace=True,
	)
	if config.subset_hvgs:
		adata = adata[:, adata.var["highly_variable"]].copy()
		adata.layers["counts"] = adata.layers["counts"]
	return adata


def preprocess_adata(adata: ad.AnnData, config: PreprocessConfig) -> ad.AnnData:
	adata = adata.copy() if config.copy_input else adata
	adata = standardize_metadata(adata)
	adata = compute_basic_qc(adata)
	adata = filter_cells_and_genes(adata, config)
	adata = normalize_and_select_hvgs(adata, config)
	return adata


def _config_from_dict(data: dict[str, Any]) -> PreprocessConfig:
	return PreprocessConfig(**{k: v for k, v in data.items() if hasattr(PreprocessConfig, k)})


def preprocess_file(input_path: str | Path, output_path: str | Path, config: PreprocessConfig) -> Path:
	input_path = Path(input_path)
	output_path = Path(output_path)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	adata = read_raw_adata(input_path)
	processed = preprocess_adata(adata, config)
	processed.write_h5ad(str(output_path))
	return output_path


def main() -> None:
	parser = argparse.ArgumentParser(description="Preprocess AnnData files for benchmarking")
	parser.add_argument("config", help="YAML preprocessing config")
	args = parser.parse_args()

	raw_cfg = load_config(args.config)
	default_cfg = _config_from_dict(raw_cfg.get("defaults", {}))
	jobs = raw_cfg.get("datasets", [])
	if not jobs:
		raise RuntimeError("No datasets found in preprocessing config")

	for job in jobs:
		job_cfg = _config_from_dict({**raw_cfg.get("defaults", {}), **job.get("preprocess", {})})
		input_path = job["input"]
		output_path = job["output"]
		print(f"Preprocessing {input_path} -> {output_path}")
		result = preprocess_file(input_path, output_path, job_cfg if job_cfg else default_cfg)
		print(f"Saved processed dataset to {result}")


if __name__ == "__main__":
	main()
