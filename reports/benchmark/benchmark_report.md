# Benchmark report: benchmark

## Comparison table

| Dataset | Method | Status | Runtime (s) | Cells | Embedding dims | Silhouette label | ARI | NMI | Silhouette batch | Failures |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pbmc3k | pca | success | 0.1556 | 2699 | 50 | — | — | — | — |  |
| pbmc3k | scvi | success | 34.5477 | 2699 | 10 | — | — | — | — |  |
| pbmc3k | scgpt | success | 90.7171 | 2699 | 512 | — | — | — | — |  |

## Dataset: pbmc3k

### pca
Status: success
Runtime: 0.16s
Cells: 2699
Embedding dims: 50

Interpretation:
- PCA provides a deterministic linear projection and is a useful reference point for comparing nonlinear embeddings.
- Quality notes: Embedding values are finite. Leiden found 8 clusters. Independent cell-type labels were unavailable or not informative.

Quality:
- Embedding values are finite.
- Leiden found 8 clusters.
- Independent cell-type labels were unavailable or not informative.
- Batch labels were unavailable or not informative.
- PCA variance ratios were captured for an additional component plot.

Metrics:
- embedding_finite: True
- embedding_dim: 50
- n_obs: 2699
- n_vars: 2000
- n_clusters: 8

Plots:
- umap: ![umap](E:/fields_of_study/bioinformatics/Projects/sc-embedding-benchmark/results/benchmark/pbmc3k/pbmc3k/pca/plots/pca_umap.png)
- explained_variance: ![explained_variance](E:/fields_of_study/bioinformatics/Projects/sc-embedding-benchmark/results/benchmark/pbmc3k/pbmc3k/pca/plots/pca_explained_variance.png)

Artifacts:
- embedding_path: E:\fields_of_study\bioinformatics\Projects\sc-embedding-benchmark\results\benchmark\pbmc3k\pbmc3k\pca\embeddings.npy
- diagnostics_path: E:\fields_of_study\bioinformatics\Projects\sc-embedding-benchmark\results\benchmark\pbmc3k\pbmc3k\pca\diagnostics.h5ad
- source_path: E:\fields_of_study\bioinformatics\Projects\sc-embedding-benchmark\data\processed\pbmc3k_processed.h5ad

### scvi
Status: success
Runtime: 34.55s
Cells: 2699
Embedding dims: 10

Interpretation:
- scVI learns a latent space directly from raw counts while modeling technical variation such as batch.
- Quality notes: Embedding values are finite. Leiden found 7 clusters. Independent cell-type labels were unavailable or not informative.

Quality:
- Embedding values are finite.
- Leiden found 7 clusters.
- Independent cell-type labels were unavailable or not informative.
- Batch labels were unavailable or not informative.

Metrics:
- embedding_finite: True
- embedding_dim: 10
- n_obs: 2699
- n_vars: 2000
- n_clusters: 7

Plots:
- umap: ![umap](E:/fields_of_study/bioinformatics/Projects/sc-embedding-benchmark/results/benchmark/pbmc3k/pbmc3k/scvi/plots/scvi_umap.png)

Artifacts:
- embedding_path: E:\fields_of_study\bioinformatics\Projects\sc-embedding-benchmark\results\benchmark\pbmc3k\pbmc3k\scvi\embeddings.npy
- diagnostics_path: E:\fields_of_study\bioinformatics\Projects\sc-embedding-benchmark\results\benchmark\pbmc3k\pbmc3k\scvi\diagnostics.h5ad
- source_path: E:\fields_of_study\bioinformatics\Projects\sc-embedding-benchmark\data\processed\pbmc3k_processed.h5ad

### scgpt
Status: success
Runtime: 90.72s
Cells: 2699
Embedding dims: 512

Interpretation:
- scGPT uses a pretrained tokenized gene representation to produce cell embeddings without retraining the foundation model.
- Quality notes: Embedding values are finite. Leiden found 6 clusters. Independent cell-type labels were unavailable or not informative.

Quality:
- Embedding values are finite.
- Leiden found 6 clusters.
- Independent cell-type labels were unavailable or not informative.
- Batch labels were unavailable or not informative.

Metrics:
- embedding_finite: True
- embedding_dim: 512
- n_obs: 2699
- n_vars: 2000
- n_clusters: 6

Plots:
- umap: ![umap](E:/fields_of_study/bioinformatics/Projects/sc-embedding-benchmark/results/benchmark/pbmc3k/pbmc3k/scgpt/plots/scgpt_umap.png)

Artifacts:
- embedding_path: E:\fields_of_study\bioinformatics\Projects\sc-embedding-benchmark\results\benchmark\pbmc3k\pbmc3k\scgpt\embeddings.npy
- diagnostics_path: E:\fields_of_study\bioinformatics\Projects\sc-embedding-benchmark\results\benchmark\pbmc3k\pbmc3k\scgpt\diagnostics.h5ad
- source_path: E:\fields_of_study\bioinformatics\Projects\sc-embedding-benchmark\data\processed\pbmc3k_processed.h5ad
