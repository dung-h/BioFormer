import os
import glob
import logging
import numpy as np
import pandas as pd
import h5py
from scipy.sparse import issparse
import scanpy as sc
from tqdm import tqdm
try:
    import cupy as cp
except ImportError:
    cp = None
    logging.error("CuPy not installed. GPU acceleration unavailable.")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/nfsshared/preprocessed/preprocess.log'),
        logging.StreamHandler()
    ]
)

def gpu_hvg_detection(file_paths, common_var_names, n_top_genes=1000, chunk_size=5000):
    """Detect highly variable genes using GPU acceleration."""
    if cp is None:
        raise ImportError("CuPy not available. Cannot perform GPU-based HVG detection.")
    
    total_cells = 0
    n_genes = len(common_var_names)
    gene_sums = cp.zeros(n_genes, dtype=cp.float32)
    gene_sum_squares = cp.zeros(n_genes, dtype=cp.float32)
    
    for f in tqdm(file_paths, desc="HVG Detection"):
        try:
            ad = sc.read_h5ad(f, backed='r')
            X = ad.raw.X
            var_names = ad.raw.var_names
            gene_to_idx = {name: i for i, name in enumerate(var_names)}
            
            indices = [gene_to_idx.get(name, None) for name in common_var_names]
            valid_indices = [i for i, idx in enumerate(indices) if idx is not None]
            dataset_indices = [idx for idx in indices if idx is not None]
            
            for start in range(0, ad.n_obs, chunk_size):
                end = min(start + chunk_size, ad.n_obs)
                X_chunk = X[start:end, dataset_indices]
                if issparse(X_chunk):
                    X_chunk = X_chunk.toarray()
                X_gpu = cp.array(X_chunk, dtype=cp.float32)
                
                chunk_sums = X_gpu.sum(axis=0)
                chunk_sum_squares = cp.square(X_gpu).sum(axis=0)
                
                full_sums = cp.zeros(n_genes, dtype=cp.float32)
                full_sum_squares = cp.zeros(n_genes, dtype=cp.float32)
                full_sums[valid_indices] = chunk_sums
                full_sum_squares[valid_indices] = chunk_sum_squares
                
                gene_sums += full_sums
                gene_sum_squares += full_sum_squares
                total_cells += (end - start)
            
            ad.file.close()
        except Exception as e:
            logging.error(f"Failed to process {f} for HVG detection: {e}")
    
    if total_cells == 0:
        raise ValueError("No valid data processed for HVG detection.")
    
    gene_means = gene_sums / total_cells
    gene_variances = (gene_sum_squares / total_cells) - cp.square(gene_means)
    gene_variances = cp.maximum(gene_variances, 1e-10)
    log_means = cp.log1p(gene_means)
    log_variances = cp.log1p(gene_variances)
    valid_mask = (gene_means > 0) & (gene_variances > 0)
    coeffs = cp.polyfit(log_means[valid_mask], log_variances[valid_mask], deg=2)
    expected_variances = cp.polyval(coeffs, log_means)
    normalized_variances = log_variances - expected_variances
    normalized_variances[~valid_mask] = -cp.inf
    hvg_indices = cp.argsort(normalized_variances)[-n_top_genes:]
    return [common_var_names[i] for i in cp.asnumpy(hvg_indices)]

def load_selected_genes(selected_genes_file, n_top_genes=1000):
    """Load selected genes from file or return None if file doesn't exist."""
    if os.path.exists(selected_genes_file):
        with open(selected_genes_file, 'r') as f:
            selected_genes = [line.strip() for line in f if line.strip()]
        if len(selected_genes) != n_top_genes:
            logging.warning(
                f"selected_genes.txt contains {len(selected_genes)} genes; expected {n_top_genes}. "
                "Recomputing HVGs."
            )
            return None
        logging.info(f"Loaded {len(selected_genes)} genes from {selected_genes_file}")
        return selected_genes
    return None

def first_pass(input_dir, output_dir, n_top_genes=1000, min_datasets=0.9):
    """Identify 1000 HVGs across selected datasets."""
    os.makedirs(output_dir, exist_ok=True)
    
    selected_datasets = [
        'dataset_85.h5ad', 'dataset_26.h5ad', 'dataset_43.h5ad', 'dataset_9.h5ad',
        'dataset_14.h5ad', 'dataset_17.h5ad', 'dataset_31.h5ad', 'dataset_54.h5ad',
        'dataset_53.h5ad', 'dataset_24.h5ad', 'dataset_59.h5ad', 'dataset_41.h5ad',
        'dataset_22.h5ad', 'dataset_37.h5ad', 'dataset_38.h5ad', 'dataset_27.h5ad',
        'dataset_34.h5ad', 'dataset_28.h5ad', 'dataset_47.h5ad', 'dataset_56.h5ad'
    ]
    files = [os.path.join(input_dir, f) for f in selected_datasets]
    files = [f for f in files if os.path.exists(f)]
    
    if not files:
        logging.error("No input files found.")
        return None
    
    logging.info(f"Processing {len(files)} datasets")
    
    var_names_list = []
    valid_files = []
    for f in files:
        try:
            ad = sc.read_h5ad(f, backed='r')
            var_names = ad.raw.var_names
            if len(var_names) == 0:
                logging.error(f"No genes found in {f}")
                ad.file.close()
                continue
            var_names_list.append(set(var_names))
            valid_files.append(f)
            logging.info(f"Loaded {f} with {len(var_names)} genes")
            ad.file.close()
        except Exception as e:
            logging.error(f"Failed to read {f}: {e}")
    
    if not var_names_list:
        logging.error("No valid datasets found.")
        return None
    
    from collections import Counter
    gene_counts = Counter()
    for var_names in var_names_list:
        gene_counts.update(var_names)
    min_count = int(len(valid_files) * min_datasets)
    common_var_names = sorted([gene for gene, count in gene_counts.items() if count >= min_count])
    
    logging.info(f"Number of genes per dataset: {[len(v) for v in var_names_list]}")
    logging.info(f"Found {len(common_var_names)} genes present in at least {min_count} datasets")
    
    if not common_var_names:
        logging.error("No common genes found even with relaxed threshold.")
        return None
    
    selected_genes_file = os.path.join(output_dir, 'selected_genes.txt')
    selected_genes = load_selected_genes(selected_genes_file, n_top_genes)
    if selected_genes is None:
        selected_genes = gpu_hvg_detection(valid_files, common_var_names, n_top_genes)
        with open(selected_genes_file, 'w') as f:
            for gene in selected_genes:
                f.write(f"{gene}\n")
        logging.info(f"Selected {len(selected_genes)} HVGs and saved to {selected_genes_file}")
    return selected_genes, valid_files

def cpu_quantile_binning(expr, num_bins=51):
    """Perform 51-bin quantile binning with zero bin."""
    binned = np.zeros_like(expr, dtype=np.uint8)
    for i in range(expr.shape[0]):
        cell = expr[i]
        nz = cell[cell > 0]
        if nz.size > 0:
            thresholds = np.quantile(nz, np.linspace(0, 1, num_bins)[1:])
            bins = np.digitize(cell, thresholds, right=True) + 1
            bins[cell == 0] = 0
            binned[i] = bins
    return binned

def preprocess_dataset(fp, selected_genes, study_id, output_dir, num_bins=51):
    """Preprocess a single dataset with padding for missing genes."""
    try:
        adata = sc.read_h5ad(fp)
        X = adata.raw.X
        var_names = adata.raw.var_names
        
        gene_to_idx = {name: i for i, name in enumerate(var_names)}
        indices = []
        valid_indices = []
        for i, gene in enumerate(selected_genes):
            idx = gene_to_idx.get(gene, None)
            if idx is not None:
                indices.append(idx)
                valid_indices.append(i)
        
        if not indices:
            logging.error(f"No selected genes found in {fp}. Skipping.")
            return
        
        matched_genes = len(indices)
        if matched_genes < len(selected_genes):
            logging.warning(
                f"Study {study_id} ({os.path.basename(fp)}) has {matched_genes} of {len(selected_genes)} "
                "selected genes; padding with zeros for missing genes"
            )
        
        X_padded = np.zeros((adata.n_obs, len(selected_genes)), dtype=np.float32)
        if indices:
            X_subset = X[:, indices].toarray() if issparse(X) else X[:, indices]
            X_padded[:, valid_indices] = X_subset
        
        binned = cpu_quantile_binning(X_padded, num_bins)
        
        adata_temp = sc.AnnData(X_padded, obs=adata.obs, var=pd.DataFrame(index=selected_genes))
        sc.pp.normalize_total(adata_temp, target_sum=1e4)
        sc.pp.log1p(adata_temp)
        expr_cont = adata_temp.X
        
        non_zero_mask = (X_padded > 0).astype(np.uint8)
        
        meta_keys = ['cell_type']
        meta = {k: adata.obs[k].values for k in meta_keys if k in adata.obs.columns}
        
        if binned.shape[1] != len(selected_genes):
            logging.error(
                f"Shape mismatch for study {study_id}: binned_expr.shape={binned.shape}, "
                f"expected (:, {len(selected_genes)})"
            )
            return
        
        out_f = os.path.join(output_dir, f"preprocessed_study{study_id}.h5")
        with h5py.File(out_f, 'w') as f:
            f.create_dataset('binned_expr', data=binned, compression='gzip', compression_opts=2)
            f.create_dataset('expr_cont', data=expr_cont, compression='gzip', compression_opts=2)
            f.create_dataset('non_zero_mask', data=non_zero_mask, compression='gzip', compression_opts=2)
            f.create_dataset('study_ids', data=np.full(adata.n_obs, study_id, dtype=np.int64), compression='gzip', compression_opts=2)
            for k, vals in meta.items():
                if vals.dtype.kind in ('O', 'U'):
                    vals_encoded = np.array([str(x).encode('utf-8') for x in vals])
                    max_len = max((len(s) for s in vals_encoded), default=1)
                    dtype_str = f'S{min(max_len, 200)}'
                    f.create_dataset(k, data=vals_encoded, dtype=dtype_str, compression='gzip', compression_opts=2)
                else:
                    f.create_dataset(k, data=vals, compression='gzip', compression_opts=2)
            f.create_dataset('var_names', data=np.array(selected_genes, dtype='S'))
        logging.info(f"Saved preprocessed data for study {study_id} to {out_f}")
    except Exception as e:
        logging.error(f"Failed to preprocess {fp}: {e}")

def main(input_dir="/nfsshared/dung/filtered_mapped", output_dir="/nfsshared/preprocessed"):
    selected_genes, files = first_pass(input_dir, output_dir, n_top_genes=1000)
    if not selected_genes:
        logging.error("Preprocessing aborted due to first pass failure.")
        return
    
    for study_id, fp in enumerate(files):
        preprocess_dataset(fp, selected_genes, study_id, output_dir)

if __name__ == "__main__":
    main()