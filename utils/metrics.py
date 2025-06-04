"""
Metrics utilities for BioFormer model evaluation.
"""

import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import (
    normalized_mutual_info_score, 
    adjusted_rand_score, 
    silhouette_score
)
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict


def compute_graph_connectivity(adata, cell_types, study_ids, n_neighbors=50):
    
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine').fit(adata.X)
    _, indices = nbrs.kneighbors(adata.X)
    
    connectivity = 0
    total = 0
    
    for i in range(len(adata)):
        cell_type = cell_types[i]
        study_id = study_ids[i]
        neighbors = indices[i][1:]  # Exclude self
        neighbor_cell_types = cell_types[neighbors]
        neighbor_study_ids = study_ids[neighbors]
        
        same_cell_type_diff_study = np.any(
            (neighbor_cell_types == cell_type) & (neighbor_study_ids != study_id)
        )
        if same_cell_type_diff_study:
            connectivity += 1
        total += 1
    
    return connectivity / total if total > 0 else 0


def compute_clustering_metrics(adata, cell_type_labels, leiden_labels=None):
    
    from sklearn.preprocessing import LabelEncoder
    
    if leiden_labels is None:
        if 'leiden' in adata.obs:
            leiden_labels = adata.obs['leiden'].values
        else:
            raise ValueError("Either provide leiden_labels or run leiden clustering first")
    
    cell_type_encoder = LabelEncoder()
    cell_type_labels_encoded = cell_type_encoder.fit_transform(cell_type_labels)
    
    nmi_score = normalized_mutual_info_score(cell_type_labels_encoded, leiden_labels)
    ari_score = adjusted_rand_score(cell_type_labels_encoded, leiden_labels)
    
    silhouette = silhouette_score(adata.X, cell_type_labels_encoded, metric='cosine')
    silhouette_normalized = (silhouette + 1) / 2
    
    graph_conn_score = compute_graph_connectivity(
        adata, cell_type_labels, adata.obs['study_id'].values, n_neighbors=50
    )
    
    avg_bio = np.mean([nmi_score, ari_score, silhouette_normalized])
    avg_batch = graph_conn_score
    
    return {
        'nmi': nmi_score,
        'ari': ari_score,
        'silhouette': silhouette_normalized,
        'graph_connectivity': graph_conn_score,
        'avg_bio': avg_bio,
        'avg_batch': avg_batch
    }


def compute_perturbation_metrics(predictions, targets, perturbations=None):
    
    correlations = []
    for i in range(predictions.shape[0]):
        mask = targets[i] > 0
        if np.sum(mask) > 0:
            corr, _ = pearsonr(predictions[i][mask], targets[i][mask])
            correlations.append(corr)
    
    result = {
        'mean_correlation': np.mean(correlations),
        'median_correlation': np.median(correlations),
        'std_correlation': np.std(correlations),
        'n_cells': len(correlations)
    }
    
    if perturbations is not None:
        group_corrs = defaultdict(list)
        for i, pert in enumerate(perturbations):
            mask = targets[i] > 0
            if np.sum(mask) > 0:
                corr, _ = pearsonr(predictions[i][mask], targets[i][mask])
                group_corrs[pert].append(corr)
                
        pert_stats = {}
        for pert, corrs in group_corrs.items():
            pert_stats[pert] = {
                'mean': np.mean(corrs),
                'median': np.median(corrs),
                'std': np.std(corrs),
                'n': len(corrs)
            }
        result['per_perturbation'] = pert_stats
    
    return result