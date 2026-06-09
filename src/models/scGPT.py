"""Flexible scGPT wrapper that uses the installed `scgpt` package.

Uses `scgpt.tasks.embed_data()` for embeddings and caches results to avoid recomputation.

Usage:
  - Provide `params` with one of:
	- `model_path`: path to model directory (used with embed_data)
	- `pretrained`: name of a pretrained model
	- `embed_fn`: explicit function path `module:func` that accepts numpy array and returns embeddings
  - Optionally provide `call` dict with kwargs passed to embed_data.
"""

import importlib
import os
from pathlib import Path
import hashlib
import json


class ScGPTWrapper:
	def __init__(self, params=None):
		self.params = params or {}
		self.model = None
		self.scgpt = None
		self.embed_fn = None
		self.cache_dir = None
		self._setup_cache()
		self._fix_windows_multiprocessing()

	def _setup_cache(self):
		"""Setup cache directory for embeddings."""
		project_root = Path(__file__).resolve().parents[2]
		self.base_cache_dir = project_root / "results" / "scgpt_cache"
		self.base_cache_dir.mkdir(parents=True, exist_ok=True)

	def _fix_windows_multiprocessing(self):
		"""Fix sched_getaffinity issue on Windows for scGPT DataLoader."""
		try:
			if not hasattr(os, "sched_getaffinity"):
				os.sched_getaffinity = lambda pid: set(range(os.cpu_count() or 1))
			
			# Also patch scGPT's cell_emb module if it's imported
			try:
				import scgpt.tasks.cell_emb as cell_emb
				cell_emb.os.sched_getaffinity = lambda pid: set()
			except Exception:
				pass  # Module not yet imported, will be patched on first use
		except Exception as e:
			print(f"Warning: Could not patch sched_getaffinity: {e}")

	def _get_cache_key(self, adata, adata_path, model_path, gene_col):
		"""Generate a cache key based on the data content and model settings."""
		obs_fingerprint = hashlib.md5("||".join(adata.obs_names.astype(str)).encode()).hexdigest()
		gene_values = adata.var[gene_col].astype(str).tolist() if gene_col in adata.var.columns else adata.var_names.astype(str).tolist()
		gene_fingerprint = hashlib.md5("||".join(gene_values).encode()).hexdigest()
		key_str = f"{adata_path}_{model_path}_{gene_col}_{adata.n_obs}_{adata.n_vars}_{obs_fingerprint}_{gene_fingerprint}"
		return hashlib.md5(key_str.encode()).hexdigest()

	def _resolve_path(self, p):
		"""Accept absolute or workspace-relative paths."""
		path = Path(p)
		if not path.is_absolute():
			# Project root: two levels up from this file (src/models)
			project_root = Path(__file__).resolve().parents[2]
			candidate = (project_root / p).resolve()
			if candidate.exists():
				return str(candidate)
		return str(path)

	def _detect_gene_col(self, adata):
		"""Auto-detect gene column name from adata.var columns."""
		# possible_names = ["gene_symbols", "gene_names", "gene_name", "genes", "gene_id", "feature_name"]
		# for col in possible_names:
		# 	if col in adata.var.columns:
		# 		print(f"Auto-detected gene column: {col}")
		# 		return col
		# # Fallback to first column
		# if len(adata.var.columns) > 0:
		# 	col = adata.var.columns[0]
		# 	print(f"Using first var column as gene column: {col}")
		# 	return col
		# raise RuntimeError("Could not find gene column in adata.var")
		adata.var["gene_symbols"] = adata.var_names.astype(str)
		return "gene_symbols"

	def fit_transform(self, adata, out_dir=None):
		"""Embed AnnData using scGPT with caching.
		
		Uses scgpt.tasks.embed_data() for embeddings and caches results to disk.
		Subsequent calls with same data/model reuse cached embeddings.
		"""
		import numpy as _np
		
		# Support explicit embed_fn if provided
		if "embed_fn" in self.params:
			mod_func = self.params["embed_fn"]
			try:
				module_name, fn_name = mod_func.split(":")
				mod = importlib.import_module(module_name)
				embed_fn = getattr(mod, fn_name)
				X = adata.X
				if hasattr(X, "toarray"):
					X = X.toarray()
				call_kwargs = self.params.get("call", {}) or {}
				embeddings = embed_fn(X, **call_kwargs)
				return _np.array(embeddings)
			except Exception as e:
				raise RuntimeError(f"embed_fn failed: {e}") from e

		# Use scgpt.tasks.embed_data() for actual embeddings
		if "model_path" not in self.params:
			raise RuntimeError(
				"scGPT requires 'model_path' in params (path to model directory). "
				"Alternatively provide 'embed_fn' (module:func) in params."
			)

		model_dir = self._resolve_path(self.params["model_path"])
		
		# Auto-detect gene column if not explicitly provided
		call_params = self.params.get("call", {}) or {}
		gene_col = call_params.get("gene_col", None)
		if gene_col is None or gene_col not in adata.var.columns:
			gene_col = self._detect_gene_col(adata)
		
		batch_size = call_params.get("batch_size", 64)
		device = call_params.get("device", "cpu")
		use_fast_transformer = call_params.get("use_fast_transformer", False)

		# Extract dataset name from adata path (e.g., "pbmc3k" from "pbmc3k_processed.h5ad")
		adata_path = str(adata.filename) if hasattr(adata, "filename") and adata.filename else "memory"
		dataset_name = Path(adata_path).stem.replace("_processed", "").replace("_annotated_benchmark", "")
		
		# Create dataset-specific cache directory
		cache_dir = self.base_cache_dir / dataset_name
		cache_dir.mkdir(parents=True, exist_ok=True)
		
		# Check cache
		cache_key = self._get_cache_key(adata, adata_path, model_dir, gene_col)
		cache_file = cache_dir / f"{cache_key}.npy"
		cache_meta = cache_dir / f"{cache_key}_meta.json"

		if cache_file.exists() and cache_meta.exists():
			print(f"Loading cached scGPT embeddings from {cache_file}")
			try:
				embeddings = _np.load(cache_file)
				with open(cache_meta, "r") as f:
					meta = json.load(f)
				if embeddings.shape[0] != adata.n_obs:
					raise ValueError(
						f"Cached embeddings have {embeddings.shape[0]} rows, but current AnnData has {adata.n_obs}."
					)
				if meta.get("shape") and list(meta["shape"]) != list(embeddings.shape):
					raise ValueError(
						f"Cached embedding metadata shape {meta.get('shape')} does not match loaded shape {list(embeddings.shape)}."
					)
				print(f"Loaded {embeddings.shape} embeddings from cache")
				return embeddings
			except Exception as e:
				print(f"Warning: Failed to load cache: {e}, recomputing...")

		# Compute embeddings using scgpt.tasks.embed_data()
		print(f"Computing scGPT embeddings using model dir: {model_dir}")
		import scgpt
		
		# Load into memory if in backed mode, then copy to avoid modifying the original
		if adata.isbacked:
			print("Loading backed AnnData into memory")
			adata_for_scgpt = adata.to_memory()
		else:
			adata_for_scgpt = adata.copy()
		
		# Patch sched_getaffinity again before calling embed_data
		try:
			import scgpt.tasks.cell_emb as cell_emb
			os.sched_getaffinity = lambda pid: set()
			cell_emb.os.sched_getaffinity = lambda pid: set()
		except Exception:
			pass

		# Call embed_data to get embeddings
		# scGPT needs gene_col to identify which column has the gene names
		print(f"Using gene_col='{gene_col}' for scGPT embedding")
		embedded_adata = scgpt.tasks.embed_data(
			adata_for_scgpt,
			model_dir,
			gene_col=gene_col,
			batch_size=batch_size,
			device=device,
			use_fast_transformer=use_fast_transformer,
			return_new_adata=True,
		)

		# Extract embeddings from result
		embeddings = _np.array(embedded_adata.X)
		
		# Cache the embeddings in dataset-specific folder
		try:
			_np.save(cache_file, embeddings)
			with open(cache_meta, "w") as f:
				json.dump({
					"dataset": dataset_name,
					"adata_path": adata_path,
					"model_dir": model_dir,
					"gene_col": gene_col,
					"shape": list(embeddings.shape),
					"dtype": str(embeddings.dtype)
				}, f, indent=2)
			print(f"Cached embeddings to {cache_file}")
		except Exception as e:
			print(f"Warning: Could not cache embeddings: {e}")

		return embeddings



