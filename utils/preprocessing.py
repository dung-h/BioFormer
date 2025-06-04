"""
Preprocessing utilities for BioFormer data.
"""

import os
import logging
import numpy as np
import pandas as pd
import scanpy as sc


def setup_logging(rank, output_dir):
    """Set up logging configuration."""
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format=f'[Rank {rank}] %(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(output_dir, f'rank_{rank}.log')),
            logging.StreamHandler()
        ]
    )


def cpu_quantile_binning(expr, num_bins=51):
    """
    Perform quantile binning with zero bin.
    
    Parameters:
    -----------
    expr : np.ndarray
        Expression matrix
    num_bins : int, default=51
        Number of bins including zero bin
        
    Returns:
    --------
    np.ndarray
        Binned expression matrix
    """
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


def preprocess_norman_dataset(adata, selected_genes, perturb_col='guide_identity', num_bins=51):
    """
    Preprocess perturbation dataset.
    
    Parameters:
    -----------
    adata : AnnData
        Annotated data matrix
    selected_genes : list
        List of genes to include
    perturb_col : str, default='guide_identity'
        Column in adata.obs containing perturbation labels
    num_bins : int, default=51
        Number of bins for expression quantization
        
    Returns:
    --------
    tuple
        (binned_expr, expr_cont, non_zero_mask, perturbations)
    """
    from scipy.sparse import issparse

    if adata.raw is not None:
        X = adata.raw.X
        var_gene_ids = adata.raw.var['gene_id']
        var_names = adata.raw.var_names
    else:
        print("[WARN] adata.raw is None, using adata.X instead")
        X = adata.X
        var_gene_ids = adata.var['gene_id']
        var_names = adata.var_names

    gene_to_idx = {gene_id: i for i, gene_id in enumerate(var_gene_ids)}

    found_genes = [g for g in selected_genes if g in gene_to_idx]
    missing_genes = [g for g in selected_genes if g not in gene_to_idx]
    print(f"[DEBUG] selected_genes count: {len(selected_genes)}")
    print(f"[DEBUG] matched genes in dataset: {len(found_genes)}")
    print(f"[DEBUG] missing genes: {len(missing_genes)}")
    
    if len(missing_genes) > 0:
        print(f"[DEBUG] Example missing genes: {missing_genes[:10]}")

    indices = []
    valid_indices = []
    for i, gene in enumerate(selected_genes):
        idx = gene_to_idx.get(gene, None)
        if idx is not None:
            indices.append(idx)
            valid_indices.append(i)

    X_padded = np.zeros((adata.n_obs, len(selected_genes)), dtype=np.float32)
    if indices:
        X_subset = X[:, indices].toarray() if issparse(X) else X[:, indices]
        X_padded[:, valid_indices] = X_subset

    print("[DEBUG] X_padded stats -- max:", X_padded.max(), "min:", X_padded.min(), 
          "nonzero:", np.count_nonzero(X_padded))

    binned = cpu_quantile_binning(X_padded, num_bins)

    adata_temp = sc.AnnData(X_padded, obs=adata.obs, var=pd.DataFrame(index=selected_genes))
    sc.pp.normalize_total(adata_temp, target_sum=1e4)
    sc.pp.log1p(adata_temp)
    expr_cont = adata_temp.X.astype(np.float32)

    non_zero_mask = (X_padded > 0).astype(np.uint8)
    print("[DEBUG] Non-zero mask stats -- sum:", non_zero_mask.sum())
    
    perturbations = adata.obs[perturb_col].astype(str).values
    
    return binned, expr_cont, non_zero_mask, perturbations


def sort_by_global_indices(embeddings, cell_types, study_ids, global_indices):
    """
    Sort data by global indices to ensure consistent ordering.
    
    Parameters:
    -----------
    embeddings : torch.Tensor or np.ndarray
        Cell embeddings
    cell_types : torch.Tensor or np.ndarray
        Cell type labels
    study_ids : torch.Tensor or np.ndarray
        Study IDs
    global_indices : torch.Tensor or np.ndarray
        Global indices for sorting
        
    Returns:
    --------
    tuple
        (sorted_embeddings, sorted_cell_types, sorted_study_ids, sorted_global_indices)
    """
    sorted_idx = global_indices.argsort()
    return (
        embeddings[sorted_idx],
        cell_types[sorted_idx],
        study_ids[sorted_idx],
        global_indices[sorted_idx]
    )