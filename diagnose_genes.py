import scanpy as sc
import pandas as pd

adata = sc.read_h5ad('notebooks/data/processed/pbmc3k_processed.h5ad')
print(f'Data shape: {adata.shape}')
print(f'Number of genes: {adata.n_vars}')
print(f'Gene column: gene_symbols')
print(f'First 20 genes: {list(adata.var["gene_symbols"].head(20))}')
print(f'Last 20 genes: {list(adata.var["gene_symbols"].tail(20))}')
print(f'Sample of middle genes: {list(adata.var["gene_symbols"].iloc[500:510])}')

# Check if there are any duplicates or weird characters
symbols = adata.var['gene_symbols'].values
print(f'Any NaN: {any(pd.isna(symbols))}')
print(f'Any duplicates: {len(symbols) != len(set(symbols))}')
print(f'Sample gene types: {[type(g) for g in symbols[:5]]}')

# Check the raw index too
print(f'adata.var.index (first 20): {list(adata.var.index[:20])}')
