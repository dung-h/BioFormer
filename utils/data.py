import os
import glob
import logging
import time
import re
import numpy as np
import pandas as pd
import h5py
import torch
from torch.utils.data import Dataset
from collections import defaultdict
from tqdm import tqdm
import scanpy as sc
import scipy.sparse
from sklearn.preprocessing import LabelEncoder

from utils.preprocessing import cpu_quantile_binning

def custom_collate_fn(batch):
    """Custom collate function to handle scalar and tensor data."""
    elem = batch[0]
    elem_type = type(elem)
    return elem_type({
        key: torch.stack([torch.as_tensor(d[key]) for d in batch])
        if isinstance(elem[key], torch.Tensor) and elem[key].dim() > 0
        else torch.tensor([d[key] for d in batch], dtype=torch.long if key in ['study_id', 'cell_type', 'global_idx', 'perturb_idx'] else elem[key].dtype)
        for key in elem
    })

class SingleCellDataset(Dataset):
    """Dataset for training with single-cell data, using merged cell types."""
    def __init__(self, data_dir, selected_genes_file=None, rank=0, num_cell_types=286):
        self.num_cell_types = num_cell_types
        self.data_files = sorted(glob.glob(os.path.join(data_dir, 'preprocessed_study*.h5')))
        if not self.data_files:
            raise ValueError(f"No HDF5 files found in {data_dir}")

        self.binned_expr = []
        self.expr_cont = []
        self.non_zero_mask = []
        self.study_ids = []
        self.cell_types = []
        self.global_indices = []
        self.cell_type_encoder = LabelEncoder()
        self.merged_cell_types = {}

        expected_shape = (1000,)
        global_idx = 0
        unique_cell_types = set()

        for f in self.data_files:
            with h5py.File(f, 'r') as h5:
                cell_types = [x.decode('utf-8') for x in h5['cell_type'][:]]
                unique_cell_types.update(cell_types)

        self._process_cell_types(unique_cell_types, num_cell_types)

        self._load_data(rank, global_idx)
        
        self._print_summary_stats(rank)

    def _process_cell_types(self, unique_cell_types, num_cell_types):
        norm_map = defaultdict(list)
        for ct in unique_cell_types:
            norm_map[self._normalize_cell_type(ct)].append(ct)
            
        self.merged_cell_types = {norm: max(names, key=len) for norm, names in norm_map.items()}
        canonical_cell_types = sorted(set(self.merged_cell_types.values()))
        self.cell_type_labels = canonical_cell_types + ['unknown'] * (num_cell_types - len(canonical_cell_types))
        self.cell_type_encoder.fit(self.cell_type_labels)
        self.cell_type_mapping = {ct: i for i, ct in enumerate(canonical_cell_types)}

    def _normalize_cell_type(self, ct):
        ct = ct.lower()
        ct = re.sub(r'[^\w\s]', '', ct)
        ct = re.sub(r'\s+', ' ', ct)
        return ct.replace('human', '').strip()

    def _load_data(self, rank, global_idx):
        for file_idx, f in enumerate(tqdm(self.data_files, desc=f"[Rank {rank}] Loading data")):
            with h5py.File(f, 'r') as h5:
                n_cells = h5['binned_expr'].shape[0]
                chunk_size = 10000
                
                for start in range(0, n_cells, chunk_size):
                    end = min(start + chunk_size, n_cells)
                    
                    binned_expr_data = h5['binned_expr'][start:end].astype(np.int32)
                    expr_cont_data = h5['expr_cont'][start:end].astype(np.float32)
                    non_zero_mask_data = h5['non_zero_mask'][start:end].astype(np.uint8)
                    study_ids_data = h5['study_ids'][start:end].astype(np.int32)
                    
                    raw_cell_types = [x.decode('utf-8') for x in h5['cell_type'][start:end]]
                    normed_cell_types = [self._normalize_cell_type(ct) for ct in raw_cell_types]
                    mapped_cell_types = [self.merged_cell_types.get(nct, 'unknown') for nct in normed_cell_types]
                    cell_type_indices = [
                        self.cell_type_mapping.get(ct, len(self.cell_type_mapping))
                        for ct in mapped_cell_types
                    ]
                    
                    self.binned_expr.extend(binned_expr_data)
                    self.expr_cont.extend(expr_cont_data)
                    self.non_zero_mask.extend(non_zero_mask_data)
                    self.study_ids.extend(study_ids_data)
                    self.cell_types.extend(cell_type_indices)
                    self.global_indices.extend([global_idx + i for i in range(end - start)])
                    
                    global_idx += end - start

        self.binned_expr = np.array(self.binned_expr, dtype=np.int32)
        self.expr_cont = np.array(self.expr_cont, dtype=np.float32)
        self.non_zero_mask = np.array(self.non_zero_mask, dtype=np.uint8)
        self.study_ids = np.array(self.study_ids, dtype=np.int32)
        self.cell_types = np.array(self.cell_types, dtype=np.int32)
        self.global_indices = np.array(self.global_indices, dtype=np.int32)

    def _print_summary_stats(self, rank):
        mem_usage = (
            self.binned_expr.nbytes +
            self.expr_cont.nbytes +
            self.non_zero_mask.nbytes +
            self.study_ids.nbytes +
            self.cell_types.nbytes
        ) / (1024 ** 3)
        
        logging.info(
            f"[Rank {rank}] Dataset loaded: {len(self.binned_expr)} cells, "
            f"{len(np.unique(self.cell_types))} unique cell types, "
            f"Memory usage: {mem_usage:.2f} GB"
        )

    def __len__(self):
        return len(self.binned_expr)

    def __getitem__(self, idx):
        return {
            'binned_expr': torch.from_numpy(self.binned_expr[idx]).long(),
            'expr_cont': torch.from_numpy(self.expr_cont[idx]).float(),
            'non_zero_mask': torch.from_numpy(self.non_zero_mask[idx]).to(torch.uint8),
            'study_id': torch.tensor(self.study_ids[idx], dtype=torch.long),
            'cell_type': torch.tensor(self.cell_types[idx], dtype=torch.long),
            'global_idx': torch.tensor(self.global_indices[idx], dtype=torch.long)
        }

class SingleCellTestDataset(Dataset):
    """Dataset for test .h5ad files, assigning study_id per file."""
    def __init__(self, data_dir, selected_genes_file, rank=0, num_cell_types=185):
        self.data_files = sorted(glob.glob(os.path.join(data_dir, '*.h5ad')))
        if not self.data_files:
            raise ValueError(f"No .h5ad files found in {data_dir}")
        
        with open(selected_genes_file, 'r') as f:
            self.selected_genes = [line.strip() for line in f if line.strip()]
        
        self.binned_expr = []
        self.expr_cont = []
        self.non_zero_mask = []
        self.study_ids = []
        self.cell_types = []
        self.global_indices = []
        self.total_cells = 0
        global_idx = 0

        for file_idx, f in enumerate(tqdm(self.data_files, desc=f"[Rank {rank}] Loading .h5ad files")):
            adata = sc.read_h5ad(f)
            if 'cell_type' not in adata.obs:
                raise ValueError(f"Missing 'cell_type' in {f}")
            if 'gene_name' not in adata.var:
                adata.var['gene_name'] = adata.var.index
            
            adata.obs['study_id'] = file_idx % 8
            
            gene_to_idx = {name: i for i, name in enumerate(adata.var['gene_name'])}
            indices = []
            valid_indices = []
            for i, gene in enumerate(self.selected_genes):
                idx = gene_to_idx.get(gene, None)
                if idx is not None:
                    indices.append(idx)
                    valid_indices.append(i)
            
            n_cells = adata.shape[0]
            X_padded = np.zeros((n_cells, len(self.selected_genes)), dtype=np.float32) 
            if indices:
                X_subset = adata.X[:, indices].toarray() if scipy.sparse.issparse(adata.X) else adata.X[:, indices]
                X_padded[:, valid_indices] = X_subset
            logging.info(f"[Rank {rank}] {f}: Matched {len(indices)}/{len(self.selected_genes)} genes")
            
            adata_temp = sc.AnnData(X_padded, obs=adata.obs, var=pd.DataFrame(index=self.selected_genes))
            sc.pp.normalize_total(adata_temp, target_sum=1e4)
            sc.pp.log1p(adata_temp)
            expr_cont = adata_temp.X.astype(np.float32)
            non_zero_mask = (X_padded > 0).astype(np.uint8)
            
            binned_expr = cpu_quantile_binning(X_padded, num_bins=51)
            
            study_ids = adata.obs['study_id'].values
            cell_types = adata.obs['cell_type'].values
            
            self.binned_expr.append(binned_expr)
            self.expr_cont.append(expr_cont)
            self.non_zero_mask.append(non_zero_mask)
            self.study_ids.append(study_ids)
            self.cell_types.append(cell_types)
            self.global_indices.extend([global_idx + i for i in range(n_cells)])
            global_idx += n_cells
            self.total_cells += n_cells
        
        self._finalize_dataset(rank)
        
        self._setup_cell_type_encoding(rank, num_cell_types)

    def _finalize_dataset(self, rank):
        self.binned_expr = np.concatenate(self.binned_expr, axis=0)
        self.expr_cont = np.concatenate(self.expr_cont, axis=0)
        self.non_zero_mask = np.concatenate(self.non_zero_mask, axis=0)
        self.study_ids = np.concatenate(self.study_ids, axis=0)
        self.cell_types = np.concatenate(self.cell_types, axis=0)
        self.global_indices = np.array(self.global_indices, dtype=np.int32)
        
        mem_usage = (
            self.binned_expr.nbytes +
            self.expr_cont.nbytes +
            self.non_zero_mask.nbytes +
            len(self.study_ids) * 4 +  # Approx for study_ids strings
            len(self.cell_types) * 10   # Approx for cell_types strings
        ) / (1024 ** 3)
        
        logging.info(
            f"[Rank {rank}] Loaded {len(self.data_files)} datasets, "
            f"{self.total_cells} cells, Memory usage: {mem_usage:.2f} GB"
        )

    def _setup_cell_type_encoding(self, rank, num_cell_types):
        self.cell_type_encoder = LabelEncoder()
        unique_cell_types = np.unique(self.cell_types)
        logging.info(f"[Rank {rank}] Found {len(unique_cell_types)} unique cell types in test data")
        
        self.cell_type_mapping = {ct: i for i, ct in enumerate(unique_cell_types)}
        self.cell_type_labels = list(unique_cell_types) + ['unknown'] * (num_cell_types - len(unique_cell_types))
        self.cell_type_encoder.fit(self.cell_type_labels)
        self.num_cell_types = num_cell_types
        
        logging.info(
            f"[Rank {rank}] Initialized model with {self.num_cell_types} cell types, "
            f"{len(unique_cell_types)} from test data"
        )

    def __len__(self):
        return len(self.global_indices)

    def __getitem__(self, idx):
        binned_expr = torch.from_numpy(self.binned_expr[idx]).long()
        expr_cont = torch.from_numpy(self.expr_cont[idx]).float()
        non_zero_mask = torch.from_numpy(self.non_zero_mask[idx]).to(torch.uint8)
        study_id = torch.tensor(self.study_ids[idx], dtype=torch.long)
        cell_type = self.cell_types[idx]
        cell_type_idx = self.cell_type_mapping.get(cell_type, len(self.cell_type_mapping))
        cell_type_idx = torch.tensor(cell_type_idx, dtype=torch.long)
        global_idx = self.global_indices[idx]
        
        return {
            'binned_expr': binned_expr,
            'expr_cont': expr_cont,
            'non_zero_mask': non_zero_mask,
            'study_id': study_id,
            'cell_type': cell_type_idx,
            'global_idx': global_idx
        }

class PerturbationDataset(Dataset):
    """Dataset for perturbation prediction."""
    def __init__(self, binned_expr, expr_cont, non_zero_mask, perturbations, label_encoder):
        self.binned_expr = binned_expr
        self.expr_cont = expr_cont
        self.non_zero_mask = non_zero_mask
        self.perturb_idx = label_encoder.transform(perturbations)

    def __len__(self):
        return len(self.binned_expr)

    def __getitem__(self, idx):
        return {
            'binned_expr': torch.from_numpy(self.binned_expr[idx]).long(),
            'expr_cont': torch.from_numpy(self.expr_cont[idx]).float(),
            'non_zero_mask': torch.from_numpy(self.non_zero_mask[idx]).to(torch.uint8),
            'perturb_idx': torch.tensor(self.perturb_idx[idx], dtype=torch.long)
        }