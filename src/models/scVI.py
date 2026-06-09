# empty
import importlib

class SCVIWrapper:
	def __init__(self, params=None):
		self.params = params or {}

	def fit_transform(self, adata, out_dir=None):
		scvi = importlib.import_module("scvi")

		# Prefer a dedicated raw counts layer produced by preprocessing.
		setup_kwargs = self.params.get("setup", {})
		if "layer" not in setup_kwargs:
			setup_kwargs["layer"] = "counts" if "counts" in adata.layers else None
		
		# Try new API (class method) first, then fall back to module-level
		try:
			scvi.model.SCVI.setup_anndata(adata, **{k: v for k, v in setup_kwargs.items() if v is not None})
		except (AttributeError, TypeError):
			try:
				scvi.data.setup_anndata(adata, **{k: v for k, v in setup_kwargs.items() if v is not None})
			except Exception:
				pass
			
		init_kwargs = self.params.get("init", {})
		model = scvi.model.SCVI(adata, **init_kwargs)
		train_kwargs = self.params.get("train", {})
		model.train(**train_kwargs)
		latent = model.get_latent_representation()
		return latent

