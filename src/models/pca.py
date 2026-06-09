from sklearn.decomposition import PCA
import numpy as np


class PCAWrapper:
	def __init__(self, params=None):
		params = params or {}
		self.n_components = params.get("n_components", 50)
		self.last_artifacts = {}

	def fit_transform(self, adata, out_dir=None):
		X = adata.X
		if hasattr(X, "toarray"):
			X = X.toarray()
		pca = PCA(n_components=self.n_components)
		Z = pca.fit_transform(X)
		self.last_artifacts = {
			"explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
			"explained_variance": pca.explained_variance_.tolist(),
		}
		return Z

