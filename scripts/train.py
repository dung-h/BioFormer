
"""
BioFormer training script with distributed data parallel support.
"""

import os
import argparse
import logging
import time
from datetime import datetime, timedelta

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from models.bioformer import BioFormer
from utils.data import SingleCellDataset, custom_collate_fn
from utils.losses import ecs_loss, bio_consistency_loss, contrastive_loss
from utils.preprocessing import setup_logging


def setup_distributed_env(rank):
    """Set up the distributed environment variables."""
    os.environ["NCCL_TIMEOUT"] = "172800000"  # 48 hours timeout
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29600"
    os.environ["NCCL_SOCKET_IFNAME"] = "lo"
    os.environ["NCCL_IGNORE_DISABLED_P2P"] = "1"
    
    logging.info(f"[Rank {rank}] NCCL_TIMEOUT set to {os.environ.get('NCCL_TIMEOUT')} ms")


def initialize_model(args, device, num_cell_types):
    """Initialize the BioFormer model."""
    model = BioFormer(
        vocab_size=args.vocab_size,
        num_studies=args.num_studies,
        num_cell_types=num_cell_types, 
        num_bins=args.num_bins,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)
    
    return model


def load_checkpoint(model, optimizer, args, device, rank):
    """Load model checkpoint if available."""
    start_epoch = 0
    if not args.resume_from_checkpoint or not os.path.exists(args.resume_from_checkpoint):
        return start_epoch
    
    checkpoint = torch.load(args.resume_from_checkpoint, map_location=device)
    state_dict = checkpoint['model_state_dict']
    model_state_dict = model.module.state_dict()
    
    for key in model_state_dict:
        if key in state_dict:
            if key == 'cell_type_embedding.weight' and state_dict[key].shape[0] != model_state_dict[key].shape[0]:
                logging.warning(
                    f"[Rank {rank}] Adjusting cell_type_embedding.weight: "
                    f"checkpoint shape {state_dict[key].shape}, model shape {model_state_dict[key].shape}"
                )
                num_types = min(state_dict[key].shape[0], model_state_dict[key].shape[0])
                model_state_dict[key][:num_types] = state_dict[key][:num_types]
            else:
                model_state_dict[key] = state_dict[key]
    
    model.module.load_state_dict(model_state_dict)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    
    logging.info(f"[Rank {rank}] Resumed from checkpoint {args.resume_from_checkpoint} at epoch {start_epoch}")
    return start_epoch


def train_epoch(model, dataloader, optimizer, scaler, args, device, rank, epoch):
    """Train for one epoch."""
    model.train()
    total_mlm_loss = 0.0
    total_cont_loss = 0.0
    total_adv_loss = 0.0
    total_ecs_loss = 0.0
    total_batch_time = 0.0
    total_steps = 0
    
    for batch in tqdm(dataloader, desc=f"[Rank {rank}] Epoch {epoch+1}/{args.epochs}", disable=(rank != 0)):
        batch_start_time = time.time()
        
        binned_expr = batch['binned_expr'].to(device, non_blocking=True)
        expr_cont = batch['expr_cont'].to(device, non_blocking=True)
        non_zero_mask = batch['non_zero_mask'].to(device, non_blocking=True)
        study_id = batch['study_id'].to(device, non_blocking=True)
        cell_type = batch['cell_type'].to(device, non_blocking=True)
        
        mask_prob = 0.15
        mask = torch.zeros_like(binned_expr, dtype=torch.bool)
        for i in range(binned_expr.size(0)):
            indices = torch.randperm(binned_expr.size(1))[:int(mask_prob * binned_expr.size(1))]
            mask[i, indices] = True
            
        effective_mask = mask & (binned_expr > 0)
        masked_input = binned_expr.clone()
        masked_input[effective_mask] = 0  # Replace masked positions with 0 (special mask token)
        
        optimizer.zero_grad(set_to_none=True)
        with autocast():
            mlm_logits, cont_pred, embeddings, adv_pred = model(
                masked_input, cell_type, study_id, non_zero_mask
            )
            
            mlm_loss = F.cross_entropy(
                mlm_logits.reshape(-1, mlm_logits.size(-1))[effective_mask.reshape(-1)],
                binned_expr.reshape(-1)[effective_mask.reshape(-1)]
            )
            
            cont_loss = F.mse_loss(cont_pred, expr_cont, reduction='none')
            cont_loss = (cont_loss * non_zero_mask).sum() / (non_zero_mask.sum() + 1e-8)
            
            adv_loss = F.cross_entropy(adv_pred, study_id)
            
            ec_loss = ecs_loss(embeddings, cell_type, margin=args.ecs_margin)
            
            total_loss = (
                args.mlm_weight * mlm_loss + 
                args.cont_weight * cont_loss + 
                args.adv_alpha * -adv_loss +  # Negative for gradient reversal
                args.ecs_weight * ec_loss
            )
        
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_mlm_loss += mlm_loss.item()
        total_cont_loss += cont_loss.item()
        total_adv_loss += adv_loss.item()
        total_ecs_loss += ec_loss.item()
        total_steps += 1
        
        batch_time = time.time() - batch_start_time
        total_batch_time += batch_time
        
    avg_mlm_loss = total_mlm_loss / total_steps
    avg_cont_loss = total_cont_loss / total_steps
    avg_adv_loss = total_adv_loss / total_steps
    avg_ecs_loss = total_ecs_loss / total_steps
    avg_batch_time = total_batch_time / total_steps
    
    return {
        'mlm_loss': avg_mlm_loss,
        'cont_loss': avg_cont_loss,
        'adv_loss': avg_adv_loss,
        'ecs_loss': avg_ecs_loss,
        'batch_time': avg_batch_time
    }


def save_checkpoint(model, optimizer, epoch, args, rank, metrics):
    """Save model checkpoint."""
    if rank != 0:
        return
        
    os.makedirs(os.path.join(args.output_dir, 'checkpoints'), exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = os.path.join(
        args.output_dir, 
        'checkpoints', 
        f'checkpoint_epoch_{epoch+1}_{timestamp}.pt'
    )
    
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.module.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'mlm_loss': metrics['mlm_loss'],
        'cont_loss': metrics['cont_loss'],
        'adv_loss': metrics['adv_loss'],
        'ecs_loss': metrics['ecs_loss'],
    }, checkpoint_path)
    
    logging.info(f"[Rank {rank}] Checkpoint saved to {checkpoint_path}")


def train(rank, world_size, args):
    """Main training function."""
    setup_distributed_env(rank)
    
    start_time = time.time()
    dist.init_process_group("nccl", rank=rank, world_size=world_size, 
                            timeout=timedelta(seconds=172800))
    logging.info(f"[Rank {rank}] DDP initialization took {time.time() - start_time:.2f} seconds")
    
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')
    
    setup_logging(rank, args.output_dir)
    
    dataset = SingleCellDataset(args.data_dir, rank=rank, num_cell_types=args.num_cell_types)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size,
        sampler=sampler,
        collate_fn=custom_collate_fn,
        num_workers=1,
        pin_memory=True
    )
    
    model = initialize_model(args, device, dataset.num_cell_types)
    model = DDP(model, device_ids=[rank])
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler()
    
    start_epoch = load_checkpoint(model, optimizer, args, device, rank)
    
    for epoch in range(start_epoch, args.epochs):
        metrics = train_epoch(model, dataloader, optimizer, scaler, args, device, rank, epoch)
        
        logging.info(
            f"[Rank {rank}] Epoch {epoch+1}/{args.epochs}: "
            f"MLM Loss={metrics['mlm_loss']:.4f}, "
            f"Cont Loss={metrics['cont_loss']:.4f}, "
            f"Adv Loss={metrics['adv_loss']:.4f}, "
            f"ECS Loss={metrics['ecs_loss']:.4f}, "
            f"Batch Time={metrics['batch_time']:.4f}s"
        )
        
        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
            save_checkpoint(model, optimizer, epoch, args, rank, metrics)
    
    if world_size > 1:
        dist.destroy_process_group()


def main():
    """Parse arguments and start training."""
    parser = argparse.ArgumentParser(description="Train BioFormer model with DDP")
    parser.add_argument('--data_dir', type=str, default="/nfsshared/preprocessed")
    parser.add_argument('--output_dir', type=str, default="/nfsshared/training")
    parser.add_argument('--resume_from_checkpoint', type=str, default=None)
    parser.add_argument('--vocab_size', type=int, default=1000)
    parser.add_argument('--num_studies', type=int, default=20)
    parser.add_argument('--num_cell_types', type=int, default=185)
    parser.add_argument('--num_bins', type=int, default=51)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--num_layers', type=int, default=12)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=74)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--save_every', type=int, default=1)
    parser.add_argument('--mlm_weight', type=float, default=1.0)
    parser.add_argument('--cont_weight', type=float, default=0.1)
    parser.add_argument('--adv_weight', type=float, default=0.1)
    parser.add_argument('--adv_alpha', type=float, default=1.0)
    parser.add_argument('--ecs_weight', type=float, default=0.1)
    parser.add_argument('--ecs_margin', type=float, default=1.0)
    parser.add_argument('--no-ddp', action='store_true', help="Run without DDP (single GPU)")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, 'train.log')),
            logging.StreamHandler()
        ]
    )
    
    if args.no_ddp:
        train(0, 1, args)
    else:
        world_size = torch.cuda.device_count()
        logging.info(f"Starting DDP training with {world_size} GPUs")
        mp.spawn(
            train,
            args=(world_size, args),
            nprocs=world_size,
            join=True
        )


if __name__ == "__main__":
    main()