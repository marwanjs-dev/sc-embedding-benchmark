# empty
import numpy as np
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
import pandas as pd
from pathlib import Path


def _guess_labels(adata):
	for col in ("labels", "cell_type", "celltype", "true_labels"):
		if col in adata.obs:
			return adata.obs[col].values
	return None


def compute_metrics(adata, embeddings, out_dir: Path):
	out_dir = Path(out_dir)
	out_dir.mkdir(parents=True, exist_ok=True)

	# compute 2D UMAP for visualization
	reducer = umap.UMAP(n_components=2, random_state=0)
	emb2 = reducer.fit_transform(embeddings)

	labels = _guess_labels(adata)

	# clustering for internal metrics
	n_clusters = len(np.unique(labels)) if labels is not None else 10
	km = KMeans(n_clusters=max(2, n_clusters), random_state=0).fit(embeddings)
	pred = km.labels_

	metrics = {}
	try:
		metrics["silhouette"] = float(silhouette_score(embeddings, pred))
	except Exception:
		metrics["silhouette"] = None

	if labels is not None:
		try:
			metrics["ari"] = float(adjusted_rand_score(labels, pred))
		except Exception:
			metrics["ari"] = None
	else:
		metrics["ari"] = None

	# save scatter plot
	fig, ax = plt.subplots(figsize=(6, 5))
	if labels is not None:
		sns.scatterplot(x=emb2[:, 0], y=emb2[:, 1], hue=labels, s=10, ax=ax, palette="tab10", legend=False)
	else:
		sns.scatterplot(x=emb2[:, 0], y=emb2[:, 1], hue=pred, s=10, ax=ax, palette="tab10", legend=False)
	ax.set_title("UMAP of embeddings")
	fig.savefig(out_dir / "umap.png", dpi=150)
	plt.close(fig)

	# write metrics to file-friendly types
	metrics_out = {k: (None if v is None else float(v)) for k, v in metrics.items()}
	return metrics_out

