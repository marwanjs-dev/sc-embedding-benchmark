from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score


@dataclass
class MethodAnalysis:
	metrics: dict[str, Any]
	quality: list[str]
	interpretation: list[str]
	plot_paths: dict[str, str]
	table_paths: dict[str, str]
	notes: list[str]


def _as_array(embeddings: Any) -> np.ndarray:
	if hasattr(embeddings, "toarray"):
		embeddings = embeddings.toarray()
	return np.asarray(embeddings)


def _safe_category(obs: pd.DataFrame, column: str, default: str) -> pd.Series:
	if column in obs.columns:
		return obs[column].astype("category")
	return pd.Series(pd.Categorical([default] * len(obs)), index=obs.index)


def _fmt(value: Any) -> str:
	if value is None:
		return "—"
	if isinstance(value, float):
		return f"{value:.4f}"
	return str(value)


def _safe_silhouette(matrix: np.ndarray, labels: pd.Series | np.ndarray) -> float | None:
	label_values = pd.Series(labels).astype(str)
	if label_values.nunique() < 2 or len(label_values) <= label_values.nunique():
		return None
	try:
		return float(silhouette_score(matrix, label_values))
	except Exception:
		return None


def _interpret_method(
	method_name: str,
	metrics: dict[str, Any],
	quality: list[str],
	summary: str,
	interpretation: str,
) -> list[str]:
	lines: list[str] = []
	if summary:
		lines.append(summary)
	if interpretation:
		lines.append(interpretation)

	ari = metrics.get("ari_cell_type")
	nmi = metrics.get("nmi_cell_type")
	label_sil = metrics.get("silhouette_cell_type")
	batch_sil = metrics.get("silhouette_batch")

	if ari is not None and nmi is not None:
		lines.append(f"Cluster agreement with labels: ARI={ari:.3f}, NMI={nmi:.3f}.")
	if label_sil is not None:
		strength = "strong" if label_sil >= 0.20 else "moderate" if label_sil >= 0.05 else "weak"
		lines.append(f"Cell-type separation is {strength} in the embedding (silhouette={label_sil:.3f}).")
	if batch_sil is not None:
		strength = "still visible" if batch_sil >= 0.20 else "present at a mild level" if batch_sil >= 0.05 else "weak"
		lines.append(f"Batch structure is {strength} (batch silhouette={batch_sil:.3f}).")
	if quality:
		lines.append("Quality notes: " + " ".join(quality[:3]))
	return lines or [f"No interpretation available for {method_name}."]


def _plot_umap(temp: ad.AnnData, colors: list[str], out_path: Path, title: str) -> None:
	if not colors:
		return
	fig = sc.pl.embedding(temp, basis="X_umap", color=colors, wspace=0.35, ncols=2, show=False, return_fig=True)
	fig.suptitle(title)
	fig.savefig(out_path, dpi=160, bbox_inches="tight")
	plt.close(fig)


def _plot_metric_bars(metrics: dict[str, Any], out_path: Path, title: str) -> None:
	keys = ["silhouette_cell_type", "ari_cell_type", "nmi_cell_type", "silhouette_batch"]
	items = [(key, metrics.get(key)) for key in keys if metrics.get(key) is not None]
	if not items:
		return
	fig, ax = plt.subplots(figsize=(7, 3.2))
	ax.bar([k.replace("_", " ") for k, _ in items], [float(v) for _, v in items])
	ax.set_ylim(-1.0, 1.0)
	ax.set_ylabel("Score")
	ax.set_title(title)
	ax.tick_params(axis="x", rotation=20)
	fig.tight_layout()
	fig.savefig(out_path, dpi=160, bbox_inches="tight")
	plt.close(fig)


def analyze_embedding(
	adata: ad.AnnData,
	embeddings: Any,
	out_dir: str | Path,
	*,
	method_name: str,
	dataset_name: str,
	label_key: str = "cell_type",
	batch_key: str = "batch",
	random_state: int = 0,
	n_neighbors: int = 15,
	summary: str = "",
	interpretation: str = "",
) -> MethodAnalysis:
	out_dir = Path(out_dir)
	plots_dir = out_dir / "plots"
	tables_dir = out_dir / "tables"
	plots_dir.mkdir(parents=True, exist_ok=True)
	tables_dir.mkdir(parents=True, exist_ok=True)

	X = _as_array(embeddings)
	if X.ndim != 2:
		raise ValueError(f"Embeddings for {method_name} must be 2D, got shape {X.shape}")
	if not np.isfinite(X).all():
		raise ValueError(f"Embeddings for {method_name} contain NaN or infinite values")

	temp = ad.AnnData(X=X)
	temp.obs = adata.obs.copy()
	temp.obs[label_key] = _safe_category(temp.obs, label_key, "unknown")
	temp.obs[batch_key] = _safe_category(temp.obs, batch_key, "batch_0")
	for column in ["total_counts", "n_genes_by_counts", "pct_counts_mt"]:
		if column in adata.obs.columns:
			temp.obs[column] = adata.obs[column].values

	metrics: dict[str, Any] = {
		"n_cells": int(temp.n_obs),
		"embedding_dim": int(temp.n_vars),
		"embedding_mean_norm": float(np.linalg.norm(X, axis=1).mean()) if temp.n_obs else None,
		"embedding_std": float(np.std(X)),
	}
	quality: list[str] = [f"Embedding finite: yes ({temp.n_obs} cells x {temp.n_vars} dims)"]
	notes: list[str] = []

	if temp.obs[label_key].nunique() > 1 and not set(temp.obs[label_key].astype(str).unique()) <= {"unknown"}:
		metrics["silhouette_cell_type"] = _safe_silhouette(X, temp.obs[label_key].astype(str))
		quality.append("Label silhouette: computed")
	else:
		notes.append("No independent cell-type labels available.")

	if temp.obs[batch_key].nunique() > 1:
		metrics["silhouette_batch"] = _safe_silhouette(X, temp.obs[batch_key].astype(str))
		quality.append("Batch silhouette: computed")
	else:
		notes.append("Only one batch detected.")

	neighbor_count = max(2, min(int(n_neighbors), max(1, temp.n_obs - 1)))
	plot_paths: dict[str, str] = {}
	table_paths: dict[str, str] = {}
	if temp.n_obs >= 3 and temp.n_vars >= 2:
		try:
			sc.pp.neighbors(temp, n_neighbors=neighbor_count, use_rep="X", random_state=random_state)
			sc.tl.umap(temp, random_state=random_state)
			sc.tl.leiden(temp, key_added="leiden", random_state=random_state)
			metrics["n_clusters"] = int(temp.obs["leiden"].nunique())
			if temp.obs[label_key].nunique() > 1 and not set(temp.obs[label_key].astype(str).unique()) <= {"unknown"}:
				metrics["ari_cell_type"] = float(adjusted_rand_score(temp.obs[label_key].astype(str), temp.obs["leiden"].astype(str)))
				metrics["nmi_cell_type"] = float(normalized_mutual_info_score(temp.obs[label_key].astype(str), temp.obs["leiden"].astype(str)))
			quality.append(f"Leiden clusters: {metrics['n_clusters']}")
			color_cols = ["leiden"]
			if label_key in temp.obs.columns:
				color_cols.append(label_key)
			if batch_key in temp.obs.columns and temp.obs[batch_key].nunique() > 1:
				color_cols.append(batch_key)
			for col in ["total_counts", "n_genes_by_counts", "pct_counts_mt"]:
				if col in temp.obs.columns:
					color_cols.append(col)
			_plot_umap(temp, color_cols, plots_dir / "umap_overview.png", f"{method_name} on {dataset_name}")
			if label_key in temp.obs.columns and temp.obs[label_key].nunique() > 1 and not set(temp.obs[label_key].astype(str).unique()) <= {"unknown"}:
				label_comp = pd.crosstab(temp.obs["leiden"], temp.obs[label_key], normalize="index")
				label_path = tables_dir / "cluster_cell_type_composition.csv"
				label_comp.to_csv(label_path)
				table_paths["cluster_cell_type_composition"] = str(label_path)
				fig, ax = plt.subplots(figsize=(max(6, label_comp.shape[1] * 0.6), max(4, label_comp.shape[0] * 0.4)))
				im = ax.imshow(label_comp.values, aspect="auto", vmin=0, vmax=1, cmap="viridis")
				ax.set_xticks(range(len(label_comp.columns)))
				ax.set_xticklabels(label_comp.columns, rotation=45, ha="right")
				ax.set_yticks(range(len(label_comp.index)))
				ax.set_yticklabels(label_comp.index)
				ax.set_title("Cluster vs cell type composition")
				fig.colorbar(im, ax=ax, label="Proportion")
				fig.tight_layout()
				fig.savefig(plots_dir / "cluster_cell_type_composition.png", dpi=160, bbox_inches="tight")
				plt.close(fig)
			if batch_key in temp.obs.columns and temp.obs[batch_key].nunique() > 1:
				batch_comp = pd.crosstab(temp.obs["leiden"], temp.obs[batch_key], normalize="index")
				batch_path = tables_dir / "cluster_batch_composition.csv"
				batch_comp.to_csv(batch_path)
				table_paths["cluster_batch_composition"] = str(batch_path)
				fig, ax = plt.subplots(figsize=(max(6, batch_comp.shape[1] * 0.6), max(4, batch_comp.shape[0] * 0.4)))
				im = ax.imshow(batch_comp.values, aspect="auto", vmin=0, vmax=1, cmap="magma")
				ax.set_xticks(range(len(batch_comp.columns)))
				ax.set_xticklabels(batch_comp.columns, rotation=45, ha="right")
				ax.set_yticks(range(len(batch_comp.index)))
				ax.set_yticklabels(batch_comp.index)
				ax.set_title("Cluster vs batch composition")
				fig.colorbar(im, ax=ax, label="Proportion")
				fig.tight_layout()
				fig.savefig(plots_dir / "cluster_batch_composition.png", dpi=160, bbox_inches="tight")
				plt.close(fig)
			_plot_metric_bars(metrics, plots_dir / "metrics_overview.png", f"{method_name} metric summary")
			plot_paths = {
				"umap_overview": str(plots_dir / "umap_overview.png"),
				"metrics_overview": str(plots_dir / "metrics_overview.png"),
			}
			if "cluster_cell_type_composition" in table_paths:
				plot_paths["cluster_cell_type_composition"] = str(plots_dir / "cluster_cell_type_composition.png")
			if "cluster_batch_composition" in table_paths:
				plot_paths["cluster_batch_composition"] = str(plots_dir / "cluster_batch_composition.png")
		except Exception as exc:
			notes.append(f"Graph/UMAP diagnostics partially unavailable: {exc}")

	if not plot_paths and temp.n_vars >= 2:
		fig, ax = plt.subplots(figsize=(6, 5))
		color = temp.obs[label_key].astype(str) if label_key in temp.obs.columns else None
		if color is not None and color.nunique() > 1 and not set(color.unique()) <= {"unknown"}:
			for group, frame in pd.DataFrame({"x": X[:, 0], "y": X[:, 1], "group": color}).groupby("group"):
				ax.scatter(frame["x"], frame["y"], s=8, label=str(group), alpha=0.8)
			ax.legend(loc="best", fontsize=7)
		else:
			ax.scatter(X[:, 0], X[:, 1], s=8, alpha=0.8)
		ax.set_xlabel("dim 1")
		ax.set_ylabel("dim 2")
		ax.set_title(f"{method_name} embedding (fallback view)")
		fig.tight_layout()
		fallback_path = plots_dir / "embedding_fallback.png"
		fig.savefig(fallback_path, dpi=160, bbox_inches="tight")
		plt.close(fig)
		plot_paths = {"embedding_fallback": str(fallback_path)}

	interpretation_lines = _interpret_method(method_name, metrics, quality, summary, interpretation)
	return MethodAnalysis(metrics=metrics, quality=quality, interpretation=interpretation_lines, plot_paths=plot_paths, table_paths=table_paths, notes=notes)
