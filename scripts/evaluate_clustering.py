
"""
Evaluation script for BioFormer clustering capabilities.
Tests the model's ability to cluster cells by type across different studies.
"""

import os
import argparse
import logging
import time
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
import scanpy as sc
import anndata
from tqdm import tqdm
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, silhouette_score

from models.bioformer import BioFormer
from utils.data import SingleCellTestDataset, custom_collate_fn
from utils.preprocessing import setup_logging, sort_by_global_indices
from utils.visualization import plot_umap, plot_attention_heatmap
from utils.metrics import compute_graph_connectivity


def evaluate(rank, world_size, args):
    """Main evaluation function."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29700"
    os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["NCCL_SOCKET_IFNAME"] = "lo"
    os.environ["NCCL_TIMEOUT"] = "3600000"  # Increase timeout to 1 hour
    os.environ["NCCL_IGNORE_DISABLED_P2P"] = "1"  # Suppress P2P warnings
    
    logging.basicConfig(level=logging.INFO)
    logging.info(f"[Rank {rank}] Starting evaluate, world_size={world_size}")
    
    if not torch.cuda.is_available():
        raise RuntimeError(f"[Rank {rank}] CUDA not available")
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')
    logging.info(f"[Rank {rank}] Using device: {device}, GPU: {torch.cuda.get_device_name(rank)}")
    
    if world_size > 1:
        logging.info(f"[Rank {rank}] Initializing process group")
        dist.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=world_size,
            timeout=torch.distributed.distributed_c10d.timedelta(seconds=3600)
        )
        logging.info(f"[Rank {rank}] Process group initialized")
    
    setup_logging(rank, args.output_dir)
    
    try:
        logging.info(f"[Rank {rank}] Loading dataset")
        dataset = SingleCellTestDataset(args.data_dir, args.selected_genes_file, rank, num_cell_types=185)
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            shuffle=False if sampler is None else None,
            num_workers=0,
            collate_fn=custom_collate_fn,
            pin_memory=True
        )
        
        logging.info(f"[Rank {rank}] Initializing model")
        model = BioFormer(
            vocab_size=args.vocab_size, # This will be 1000 from args
            num_cell_types=185, # Note: Mismatch with dataset's 185, handled in checkpoint.
            num_bins=args.num_bins,
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            dropout=args.dropout
        ).to(device)
        
        logging.info(f"[Rank {rank}] Loading checkpoint")
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        state_dict = checkpoint['model_state_dict']
        model_state_dict = model.state_dict()
        
        for key in model_state_dict:
            if key in state_dict:
                if key == 'cell_type_embedding.weight' and state_dict[key].shape != model_state_dict[key].shape:
                    logging.warning(f"[Rank {rank}] Adjusting cell_type_embedding.weight: checkpoint shape {state_dict[key].shape}, model shape {model_state_dict[key].shape}")
                    num_types = min(state_dict[key].shape[0], model_state_dict[key].shape[0])
                    model_state_dict[key][:num_types] = state_dict[key][:num_types]
                else:
                    model_state_dict[key] = state_dict[key]
        
        model.load_state_dict(model_state_dict)
        if world_size > 1:
            model = DDP(model, device_ids=[rank])
        logging.info(f"[Rank {rank}] Loaded checkpoint from {args.checkpoint_path}")
        
        model.eval()
        embeddings = []
        cell_types = []
        study_ids = []
        global_indices = []
        
        logging.info(f"[Rank {rank}] Starting evaluation")
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"[Rank {rank}] Evaluating", disable=(rank != 0)):
                binned_expr = batch['binned_expr'].to(device, non_blocking=True)
                non_zero_mask = batch['non_zero_mask'].to(device, non_blocking=True)
                cell_type = batch['cell_type'].to(device, non_blocking=True)
                study_id = batch['study_id'].to(device, non_blocking=True)
                global_idx = batch['global_idx'].to(device, non_blocking=True)
                
                with autocast():
                    if args.visualize_attention:
                        _, _, output, attention = model(
                            binned_expr=binned_expr,
                            cell_type=cell_type,
                            non_zero_mask=non_zero_mask,
                            return_attention=True
                        )
                    else:
                        _, _, output = model(
                            binned_expr=binned_expr,
                            cell_type=cell_type,
                            non_zero_mask=non_zero_mask
                        )

                embeddings.append(output[:,0].cpu())  # Keep on CUDA
                cell_types.append(cell_type)
                study_ids.append(study_id)
                global_indices.append(global_idx)
        
        torch.cuda.empty_cache()
        embeddings = torch.cat(embeddings, dim=0)
        cell_types = torch.cat(cell_types, dim=0)
        study_ids = torch.cat(study_ids, dim=0)
        global_indices = torch.cat(global_indices, dim=0)
        
        if world_size > 1:
            logging.info(f"[Rank {rank}] Gathering results")
            embeddings = embeddings.contiguous()
            cell_types = cell_types.contiguous()
            study_ids = study_ids.contiguous()
            global_indices = global_indices.contiguous()
            
            logging.info(f"[Rank {rank}] embeddings shape: {embeddings.shape}, device: {embeddings.device}")
            logging.info(f"[Rank {rank}] cell_types shape: {cell_types.shape}, device: {cell_types.device}")
            logging.info(f"[Rank {rank}] study_ids shape: {study_ids.shape}, device: {cell_types.device}")
            logging.info(f"[Rank {rank}] global_indices shape: {global_indices.shape}, device: {cell_types.device}")
            
            total_size = len(dataset)
            logging.info(f"[Rank {rank}] Total dataset size: {total_size}")
            embeddings_per_rank = [torch.zeros((total_size // world_size + (1 if i < total_size % world_size else 0), args.d_model), device=device) for i in range(world_size)]
            cell_types_per_rank = [torch.zeros(total_size // world_size + (1 if i < total_size % world_size else 0), dtype=torch.long, device=device) for i in range(world_size)]
            study_ids_per_rank = [torch.zeros(total_size // world_size + (1 if i < total_size % world_size else 0), dtype=torch.long, device=device) for i in range(world_size)]
            global_indices_per_rank = [torch.zeros(total_size // world_size + (1 if i < total_size % world_size else 0), dtype=torch.long, device=device) for i in range(world_size)]
            
            dist.barrier()
            
            dist.all_gather(embeddings_per_rank, embeddings)
            dist.all_gather(cell_types_per_rank, cell_types)
            dist.all_gather(study_ids_per_rank, study_ids)
            dist.all_gather(global_indices_per_rank, global_indices)
            
            embeddings = torch.cat(embeddings_per_rank, dim=0).cpu()
            cell_types = torch.cat(cell_types_per_rank, dim=0).cpu()
            study_ids = torch.cat(study_ids_per_rank, dim=0).cpu()
            global_indices = torch.cat(global_indices_per_rank, dim=0).cpu()
        else:
            embeddings = embeddings.cpu()
            cell_types = cell_types.cpu()
            study_ids = study_ids.cpu()
            global_indices = global_indices.cpu()
        
        if rank == 0:
            cell_type_labels = dataset.cell_type_encoder.inverse_transform(cell_types.numpy())
            
            adata = anndata.AnnData(
                X=embeddings.numpy(),
                obs={
                    'cell_type': cell_type_labels,
                    'study_id': study_ids.numpy().astype(str)
                }
            )
            
            sc.pp.neighbors(adata, use_rep='X', n_neighbors=15, metric='cosine')
            sc.tl.leiden(adata, resolution=1.0, key_added='leiden')
            
            result_dir = os.path.join(args.output_dir, 'test_results')
            os.makedirs(result_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_umap(embeddings.numpy(), cell_type_labels, adata.obs['study_id'].values, 'study_ids', 'cell_types', result_dir, timestamp, 'BioFormer: Study ID (color) and Cell Type (marker)')
            plot_umap(embeddings.numpy(), cell_type_labels, adata.obs['study_id'].values, 'cell_types', 'study_ids', result_dir, timestamp, 'BioFormer: Cell Type (color) and Study ID (marker)')
            logging.info(f"[Rank {rank}] Saved UMAP plots to {result_dir}")
            
            try:
                leiden_labels = adata.obs['leiden'].values
                cell_type_labels_encoded = LabelEncoder().fit_transform(cell_type_labels)
                nmi_score = normalized_mutual_info_score(cell_type_labels_encoded, leiden_labels)
                ari_score = adjusted_rand_score(cell_type_labels_encoded, leiden_labels)
                
                silhouette = silhouette_score(adata.X, cell_type_labels_encoded, metric='cosine')
                silhouette_normalized = (silhouette + 1) / 2
                
                graph_conn_score = compute_graph_connectivity(
                    adata, cell_type_labels, adata.obs['study_id'].values, n_neighbors=50
                )
                
                avg_bio = np.mean([nmi_score, ari_score, silhouette_normalized])
                avg_batch = graph_conn_score
                
                logging.info(f"[Rank {rank}] NMI Score: {nmi_score:.4f}")
                logging.info(f"[Rank {rank}] ARI Score: {ari_score:.4f}")
                logging.info(f"[Rank {rank}] Silhouette Score: {silhouette_normalized:.4f}")
                logging.info(f"[Rank {rank}] Graph Connectivity Score: {graph_conn_score:.4f}")
                logging.info(f"[Rank {rank}] AvgBio Score: {avg_bio:.4f}")
                logging.info(f"[Rank {rank}] AvgBatch Score: {avg_batch:.4f}")
                
                with open(os.path.join(result_dir, 'metrics.txt'), 'w') as f:
                    f.write(f"NMI Score: {nmi_score:.4f}\n")
                    f.write(f"ARI Score: {ari_score:.4f}\n")
                    f.write(f"Silhouette Score: {silhouette_normalized:.4f}\n")
                    f.write(f"Graph Connectivity Score: {graph_conn_score:.4f}\n")
                    f.write(f"AvgBio Score: {avg_bio:.4f}\n")
                    f.write(f"AvgBatch Score: {avg_batch:.4f}\n")
                logging.info(f"[Rank {rank}] Saved metrics to {os.path.join(result_dir, 'metrics.txt')}")
            except Exception as e:
                logging.warning(f"[Rank {rank}] Failed to compute metrics: {str(e)}. Skipping metrics computation.")
    
    finally:
        if world_size > 1:
            if dist.is_initialized():
                dist.barrier()
                dist.destroy_process_group()
        torch.cuda.empty_cache()
        logging.info(f"[Rank {rank}] Evaluation complete with cleanup")
        mp.set_start_method('spawn', force=True)


def main():
    parser = argparse.ArgumentParser(description="Test BioFormer model with DDP")
    parser.add_argument('--data_dir', type=str, default="/home/tripham/BioFormer/test_clustering/test3")
    parser.add_argument('--selected_genes_file', type=str, default="/nfsshared/preprocessed/selected_genes.txt")
    parser.add_argument('--output_dir', type=str, default="/nfsshared/training1")
    parser.add_argument('--checkpoint_path', type=str, default="/nfsshared/training/checkpoints/checkpoint_epoch_1_20250604_065624.pt")
    parser.add_argument('--vocab_size', type=int, default=1000) 
    parser.add_argument('--num_studies', type=int, default=20)
    parser.add_argument('--num_bins', type=int, default=51)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--num_layers', type=int, default=12)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--mlm_weight', type=float, default=1.0)
    parser.add_argument('--cont_weight', type=float, default=0.1)
    parser.add_argument('--adv_weight', type=float, default=0.1)
    parser.add_argument('--adv_alpha', type=float, default=1.0)
    parser.add_argument('--ecs_weight', type=float, default=0.1)
    parser.add_argument('--ecs_margin', type=float, default=1.0)
    parser.add_argument('--no-ddp', action='store_true', help="Run without DDP (single GPU)")
    parser.add_argument('--visualize_attention', default=False, help="Enable attdention extraction during evaluation")
  
    args = parser.parse_args()
    
    if args.no_ddp:
        logging.basicConfig(level=logging.INFO)
        logging.info("Running without DDP on single GPU")
        evaluate(0, 1, args)
    else:
        world_size = torch.cuda.device_count()
        if world_size < 1:
            raise RuntimeError(f"DDP requires at least 1 GPU, found {world_size}")
        mp.spawn(evaluate, args=(world_size, args), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()