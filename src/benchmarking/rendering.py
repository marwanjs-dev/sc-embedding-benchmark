from __future__ import annotations

from pathlib import Path
from typing import Any

from .specs import BenchmarkRunResult, MethodRunResult


def _fmt(value: Any) -> str:
	if value is None:
		return "—"
	if isinstance(value, float):
		return f"{value:.4f}"
	return str(value)


def _write_markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
	head = "| " + " | ".join(columns) + " |"
	sep = "| " + " | ".join(["---"] * len(columns)) + " |"
	lines = [head, sep]
	for row in rows:
		lines.append("| " + " | ".join(_fmt(row.get(column)) for column in columns) + " |")
	return "\n".join(lines)


def _result_row(dataset_name: str, result: MethodRunResult) -> dict[str, Any]:
	return {
		"Dataset": dataset_name,
		"Method": result.method_name,
		"Status": result.status,
		"Runtime (s)": result.runtime_seconds,
		"Cells": result.n_obs,
		"Embedding dims": result.embedding_dim,
		"Silhouette label": result.metrics.get("silhouette_cell_type"),
		"ARI": result.metrics.get("ari_cell_type"),
		"NMI": result.metrics.get("nmi_cell_type"),
		"Silhouette batch": result.metrics.get("silhouette_batch"),
		"Failures": result.failure[:120] if result.failure else "",
	}


def _method_section(result: MethodRunResult) -> list[str]:
	lines = [f"### {result.method_name}", f"Status: {result.status}", f"Runtime: {result.runtime_seconds:.2f}s", f"Cells: {result.n_obs}", f"Embedding dims: {result.embedding_dim}", ""]
	if result.interpretation:
		lines.append("Interpretation:")
		for line in result.interpretation:
			lines.append(f"- {line}")
		lines.append("")
	if result.quality:
		lines.append("Quality:")
		for line in result.quality:
			lines.append(f"- {line}")
		lines.append("")
	if result.metrics:
		lines.append("Metrics:")
		for key, value in result.metrics.items():
			lines.append(f"- {key}: {_fmt(value)}")
		lines.append("")
	if result.plot_paths:
		lines.append("Plots:")
		for key, path in result.plot_paths.items():
			lines.append(f"- {key}: ![{key}]({Path(path).as_posix()})")
		lines.append("")
	# if result.artifacts:
	# 	lines.append("Artifacts:")
	# 	for key, value in result.artifacts.items():
	# 		lines.append(f"- {key}: {value}")
	# 	lines.append("")
	if result.failure:
		lines.append("Failures:")
		lines.append("```text")
		lines.append(result.failure)
		if result.traceback:
			lines.append(result.traceback)
		lines.append("```")
		lines.append("")
	return lines


def write_benchmark_report(run_result: BenchmarkRunResult) -> Path:
	output_dir = Path(run_result.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)
	report_path = output_dir / "benchmark_report.md"

	rows: list[dict[str, Any]] = []
	for dataset in run_result.datasets:
		for result in dataset.methods:
			rows.append(_result_row(dataset.name, result))
	lines: list[str] = []
	lines.append(f"# Benchmark report:")
	lines.append("")
	lines.append("## Comparison table")
	lines.append("")
	lines.append(_write_markdown_table(rows, ["Dataset", "Method", "Status", "Runtime (s)", "Cells", "Embedding dims", "Silhouette label", "ARI", "NMI", "Silhouette batch", "Failures"]))
	lines.append("")

	for dataset in run_result.datasets:
		lines.append(f"## Dataset: {dataset.name}")
		if dataset.notes:
			lines.append(dataset.notes)
		lines.append("")
		for result in dataset.methods:
			lines.extend(_method_section(result))

	report_path.write_text("\n".join(lines), encoding="utf-8")
	return report_path
