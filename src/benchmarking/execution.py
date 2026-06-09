from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any
import importlib
import json
import traceback

import numpy as np

from src.preprocessing import PreprocessConfig, preprocess_file, standardize_metadata

from .diagnostics import analyze_embedding
from .specs import BenchmarkConfig, BenchmarkRunResult, DatasetRunResult, DatasetSpec, MethodRunResult, MethodSpec


def _resolve_callable(path: str):
	module_name, attr_name = path.split(":")
	return getattr(importlib.import_module(module_name), attr_name)


def _resolve_path(path: str | None, project_root: Path) -> Path | None:
	if path is None:
		return None
	p = Path(path)
	return p if p.is_absolute() else (project_root / p).resolve()


def _build_preprocess_config(raw: dict[str, Any] | None) -> PreprocessConfig:
	raw = raw or {}
	return PreprocessConfig(
		min_genes=int(raw.get("min_genes", 200)),
		min_cells=int(raw.get("min_cells", 3)),
		max_pct_mt=float(raw.get("max_pct_mt", 20.0)),
		min_counts=raw.get("min_counts"),
		n_top_genes=int(raw.get("n_top_genes", 2000)),
		hvg_flavor=raw.get("hvg_flavor", "cell_ranger"),
		subset_hvgs=bool(raw.get("subset_hvgs", True)),
		make_unique=bool(raw.get("make_unique", True)),
		copy_input=bool(raw.get("copy_input", False)),
	)


def _prepare_dataset(dataset: DatasetSpec, project_root: Path) -> tuple[Any, Path]:
	load_path = _resolve_path(dataset.path, project_root)
	if load_path is None:
		raise ValueError(f"Dataset '{dataset.name}' is missing a path")
	if dataset.preprocess:
		pre_cfg = dataset.preprocess.copy()
		input_path = _resolve_path(pre_cfg.pop("input", None), project_root)
		output_path = _resolve_path(pre_cfg.pop("output", None), project_root)
		if input_path is None or output_path is None:
			raise ValueError(f"Dataset '{dataset.name}' preprocess config requires input and output")
		preprocess_file(input_path, output_path, _build_preprocess_config(pre_cfg))
		load_path = output_path
	loader = _resolve_callable(dataset.loader)
	adata = loader(str(load_path))
	adata = standardize_metadata(adata)
	adata.var_names_make_unique()
	if "gene_symbols" not in adata.var.columns:
		adata.var["gene_symbols"] = adata.var_names.astype(str)
	return adata, load_path


def _sanitize_embedding(embedding: Any) -> np.ndarray:
	if hasattr(embedding, "toarray"):
		embedding = embedding.toarray()
	return np.asarray(embedding)


def _run_method(adata, dataset: DatasetSpec, method: MethodSpec, result_dir: Path, config: BenchmarkConfig) -> MethodRunResult:
	result_dir.mkdir(parents=True, exist_ok=True)
	start = perf_counter()
	model = None
	try:
		wrapper_cls = _resolve_callable(method.module)
		model = wrapper_cls(method.params)
		embeddings = _sanitize_embedding(model.fit_transform(adata.copy(), out_dir=result_dir))
		if embeddings.ndim != 2:
			raise ValueError(f"Method {method.name} returned embeddings with shape {embeddings.shape}")
		embedding_path = result_dir / "embeddings.npy"
		np.save(embedding_path, embeddings)
		extra = getattr(model, "last_artifacts", {}) or {}
		analysis = analyze_embedding(
			adata,
			embeddings,
			result_dir,
			method_name=method.name,
			dataset_name=dataset.name,
			label_key="cell_type",
			batch_key="batch",
			random_state=config.seed,
			n_neighbors=config.n_neighbors,
			summary=method.summary,
			interpretation=method.interpretation,
		)
		runtime = perf_counter() - start
		return MethodRunResult(
			dataset_name=dataset.name,
			method_name=method.name,
			module=method.module,
			class_name=model.__class__.__name__ if model is not None else method.module,
			status="success",
			runtime_seconds=runtime,
			output_dir=str(result_dir),
			embedding_path=str(embedding_path),
			metrics=analysis.metrics,
			plot_paths=analysis.plot_paths,
			table_paths=analysis.table_paths,
			quality=analysis.quality,
			interpretation=analysis.interpretation,
			notes=method.notes + analysis.notes,
			extra=extra,
			n_obs=int(adata.n_obs),
			n_vars=int(adata.n_vars),
			embedding_dim=int(embeddings.shape[1]),
		)
	except Exception as exc:
		runtime = perf_counter() - start
		err_path = result_dir / "failure.txt"
		err_path.write_text(traceback.format_exc(), encoding="utf-8")
		return MethodRunResult(
			dataset_name=dataset.name,
			method_name=method.name,
			module=method.module,
			class_name=model.__class__.__name__ if model is not None else method.module,
			status="failed",
			runtime_seconds=runtime,
			output_dir=str(result_dir),
			error=str(exc),
			traceback=traceback.format_exc(),
			notes=method.notes,
			extra={"failure_path": str(err_path)},
			n_obs=int(adata.n_obs),
			n_vars=int(adata.n_vars),
            embedding_dim=None,
            failure=traceback.format_exc()[:120] if traceback.format_exc() else None
		)


def run_benchmark(spec: BenchmarkConfig, project_root: Path) -> BenchmarkRunResult:
	output_dir = Path(spec.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)
	run = BenchmarkRunResult(created_at=datetime.now().isoformat(timespec="seconds"), output_dir=str(output_dir), config=spec, datasets=[])
	for dataset in spec.datasets:
		dataset_dir = output_dir / dataset.name
		dataset_dir.mkdir(parents=True, exist_ok=True)
		adata, loaded_path = _prepare_dataset(dataset, project_root)
		dataset_result = DatasetRunResult(
			name=dataset.name,
			source_path=str(loaded_path),
			output_dir=str(dataset_dir),
			shape=(int(adata.n_obs), int(adata.n_vars)),
			label_key="cell_type",
			batch_key="batch",
			notes=dataset.notes,
			methods=[],
		)
		for method in spec.methods:
			dataset_result.methods.append(_run_method(adata, dataset, method, dataset_dir / method.name, spec))
		run.datasets.append(dataset_result)
	with open(output_dir / "benchmark_summary.json", "w", encoding="utf-8") as handle:
		json.dump({"created_at": run.created_at, "output_dir": run.output_dir, "config": asdict(spec), "datasets": [asdict(dataset) for dataset in run.datasets]}, handle, indent=2, default=str)
	return run
