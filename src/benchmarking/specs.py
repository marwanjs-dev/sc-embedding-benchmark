from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DatasetSpec:
	name: str
	path: str | None = None
	loader: str = "src.utils.io:load_h5ad"
	preprocess: dict[str, Any] | None = None
	label_key: str | None = "labels"
	batch_key: str | None = "batch"
	notes: str = ""


@dataclass
class MethodSpec:
	name: str
	module: str
	params: dict[str, Any] = field(default_factory=dict)
	summary: str = ""
	interpretation: str = ""
	notes: list[str] = field(default_factory=list)


@dataclass
class BenchmarkConfig:
	output_dir: str = "results/benchmark"
	project_root: str | None = None
	seed: int = 0
	n_neighbors: int = 15
	jobs: int = 1
	datasets: list[DatasetSpec] = field(default_factory=list)
	methods: list[MethodSpec] = field(default_factory=list)


@dataclass
class MethodRunResult:
	dataset_name: str
	method_name: str
	module: str
	class_name: str
	status: str
	runtime_seconds: float | None = None
	output_dir: str = ""
	embedding_path: str | None = None
	metrics: dict[str, Any] = field(default_factory=dict)
	plot_paths: dict[str, str] = field(default_factory=dict)
	table_paths: dict[str, str] = field(default_factory=dict)
	quality: list[str] = field(default_factory=list)
	interpretation: list[str] = field(default_factory=list)
	notes: list[str] = field(default_factory=list)
	error: str | None = None
	traceback: str | None = None
	extra: dict[str, Any] = field(default_factory=dict)
	n_obs: int | None = None
	n_vars: int | None = None
	embedding_dim: int | None = None
	failure: str | None = None
 
 
 		# "Dataset": dataset_name,
		# "Method": result.method_name,
		# "Status": result.status,
		# "Runtime (s)": result.runtime_seconds,
		# "Cells": result.n_obs,
		# "Embedding dims": result.embedding_dim,
		# "Silhouette label": result.metrics.get("silhouette_cell_type"),
		# "ARI": result.metrics.get("ari_cell_type"),
		# "NMI": result.metrics.get("nmi_cell_type"),
		# "Silhouette batch": result.metrics.get("silhouette_batch"),
		# "Failures": result.failure[:120] if result.failure else "",


@dataclass
class DatasetRunResult:
	name: str
	source_path: str | None
	output_dir: str
	shape: tuple[int, int] | None = None
	label_key: str | None = None
	batch_key: str | None = None
	notes: str = ""
	methods: list[MethodRunResult] = field(default_factory=list)


@dataclass
class BenchmarkRunResult:
	created_at: str
	output_dir: str
	config: BenchmarkConfig
	datasets: list[DatasetRunResult] = field(default_factory=list)
 
# @dataclass
# class MethodResult:
#     dataset_name: str
#     method_name: str
#     module: str
#     class_name: str
#     status: str
#     runtime_seconds: float | None = None
#     output_dir: str = ""
#     embedding_path: str | None = None
#     metrics: dict[str, Any] = field(default_factory=dict)
#     plot_paths: dict[str, str] = field(default_factory=dict)
#     table_paths: dict[str, str] = field(default_factory=dict)
#     quality: list[str] = field(default_factory=list)
#     interpretation: list[str] = field(default_factory=list)
#     notes: list[str] = field(default_factory=list)
#     error: str | None = None
#     traceback: str | None = None
#     extra: dict[str, Any] = field(default_factory=dict)
#     n_obs: int | None = None
#     n_vars: int | None = None
#     embedding_dim: int | None = None
    



def _maybe_dict(value: Any) -> dict[str, Any]:
	return value if isinstance(value, dict) else {}


def _dataset_from_mapping(name: str, raw: dict[str, Any]) -> DatasetSpec:
	return DatasetSpec(
		name=name,
		path=raw.get("path"),
		loader=raw.get("loader", "src.utils.io:load_h5ad"),
		preprocess=_maybe_dict(raw.get("preprocess")) or None,
		label_key=raw.get("label_key", "labels"),
		batch_key=raw.get("batch_key", "batch"),
		notes=raw.get("notes", ""),
	)


def _method_from_mapping(name: str, raw: dict[str, Any]) -> MethodSpec:
	return MethodSpec(
		name=name,
		module=raw["module"],
		params=_maybe_dict(raw.get("params")),
		summary=raw.get("summary", ""),
		interpretation=raw.get("interpretation", ""),
		notes=list(raw.get("notes", []) or []),
	)


def load_benchmark_config(path: str | Path) -> BenchmarkConfig:
	with open(path, "r", encoding="utf-8") as handle:
		raw = yaml.safe_load(handle) or {}

	benchmark_cfg = _maybe_dict(raw.get("benchmark"))
	output_dir = benchmark_cfg.get("output_dir", raw.get("output_dir", "results/benchmark"))
	seed = int(benchmark_cfg.get("seed", raw.get("seed", 0)))
	n_neighbors = int(benchmark_cfg.get("n_neighbors", raw.get("n_neighbors", 15)))
	jobs = int(benchmark_cfg.get("jobs", raw.get("jobs", 1)))

	datasets: list[DatasetSpec] = []
	methods: list[MethodSpec] = []

	if "experiments" in raw:
		for index, exp in enumerate(raw.get("experiments", [])):
			exp_name = exp.get("name", f"experiment_{index}")
			dataset_cfg = _maybe_dict(exp.get("dataset"))
			model_cfg = _maybe_dict(exp.get("model"))
			datasets.append(_dataset_from_mapping(exp_name, dataset_cfg))
			methods.append(
				MethodSpec(
					name=exp_name,
					module=model_cfg["module"],
					params=_maybe_dict(model_cfg.get("params")),
					summary=model_cfg.get("summary", ""),
					interpretation=model_cfg.get("interpretation", ""),
					notes=list(model_cfg.get("notes", []) or []),
				)
			)
	else:
		for name, dataset_raw in (raw.get("datasets", {}) or {}).items():
			datasets.append(_dataset_from_mapping(name, _maybe_dict(dataset_raw)))
		for name, method_raw in (raw.get("methods", {}) or {}).items():
			methods.append(_method_from_mapping(name, _maybe_dict(method_raw)))

	if not datasets:
		raise RuntimeError("No datasets defined in benchmark config")
	if not methods:
		raise RuntimeError("No methods defined in benchmark config")

	project_root = benchmark_cfg.get("project_root", raw.get("project_root"))
	return BenchmarkConfig(
		output_dir=output_dir,
		project_root=project_root,
		seed=seed,
		n_neighbors=n_neighbors,
		jobs=jobs,
		datasets=datasets,
		methods=methods,
	)
