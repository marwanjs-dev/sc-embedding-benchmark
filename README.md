# sc-embedding-benchmark

This repository now includes a modular benchmark runner that compares PCA, scVI, and scGPT on the same dataset and writes a single markdown report with a comparison table, method sections, plots, quality notes, and failure traces when a method fails.

Run the benchmark with:

```bash
python src/benchmark.py configs/benchmark_example.yaml
```

The report is written under the configured benchmark output directory, along with per-method plots and tables. To add a new method later, add one more entry under `methods` in the benchmark config and point it at a wrapper class that implements `fit_transform(adata, out_dir=None)`.
