
"""
BioFormer perturbation model training script.
Trains a transformer-based model to predict gene expression after perturbation.
"""

import os
import argparse
import logging
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import scanpy as sc

from models.perturbation import BioFormerPerturb
from utils.preprocessing import preprocess_norman_dataset
from utils.data import PerturbationDataset
from utils.metrics import compute_perturbation_metrics
from utils.utils import load_model_with_mismatch


def evaluate(model, dataloader):
    """
    Evaluate the model's performance on a dataset.
    Returns the mean Pearson correlation.
    """
    model.eval()
    all_preds, all_targets, all_perturbs = [], [], []
    
    with torch.no_grad():
        for batch in dataloader:
            binned_expr = batch['binned_expr'].cuda()
            expr_cont = batch['expr_cont'].cuda()
            non_zero_mask = batch['non_zero_mask'].cuda()
            perturb_idx = batch['perturb_idx'].cuda()
            
            with autocast():
                cont_pred = model(binned_expr, perturb_idx, non_zero_mask)

            all_preds.append(cont_pred.cpu().numpy())
            all_targets.append(expr_cont.cpu().numpy())
            all_perturbs.append(perturb_idx.cpu().numpy())

    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    perturbs = np.concatenate(all_perturbs)
    
    metrics = compute_perturbation_metrics(preds, targets, perturbs)
    print(f"[Eval] Mean Pearson Correlation = {metrics['mean_correlation']:.4f}")

    if 'per_perturbation' in metrics:
        for k in sorted(metrics['per_perturbation'].keys()):
            pert_stat = metrics['per_perturbation'][k]
            print(f"[Eval] Perturb {k}: Mean corr = {pert_stat['mean']:.4f} (n={pert_stat['n']})")

    return metrics['mean_correlation']


def train(args):
    """Main training function."""
    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(args.output_dir, 'train.log'),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    logging.info(f"Loading data from {args.h5ad}")
    adata = sc.read_h5ad(args.h5ad)
    selected_genes = [line.strip() for line in open(args.hvg_file)]
    logging.info(f"Loaded {len(selected_genes)} selected genes from {args.hvg_file}")
    
    binned_expr, expr_cont, non_zero_mask, perturbations = preprocess_norman_dataset(
        adata, selected_genes, perturb_col=args.perturb_col
    )
    logging.info(f"Data preprocessed: {len(perturbations)} samples with {len(selected_genes)} genes")

    label_encoder = LabelEncoder()
    label_encoder.fit(perturbations)
    logging.info(f"Found {len(label_encoder.classes_)} unique perturbations")

    indices = np.arange(len(binned_expr))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.2, stratify=perturbations, random_state=42
    )
    logging.info(f"Split data: {len(train_idx)} train, {len(test_idx)} test samples")

    train_dataset = PerturbationDataset(
        binned_expr[train_idx], expr_cont[train_idx],
        non_zero_mask[train_idx], perturbations[train_idx], label_encoder
    )
    test_dataset = PerturbationDataset(
        binned_expr[test_idx], expr_cont[test_idx],
        non_zero_mask[test_idx], perturbations[test_idx], label_encoder
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = BioFormerPerturb(
        vocab_size=len(selected_genes),
        num_perturbs=len(label_encoder.classes_),
        num_bins=args.num_bins,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).cuda()
    logging.info(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")

    if args.pretrained_ckpt:
        logging.info(f"Loading pretrained checkpoint from {args.pretrained_ckpt}")
        load_model_with_mismatch(model, args.pretrained_ckpt)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = GradScaler()

    np.savez_compressed(
        os.path.join(args.output_dir, "test_data.npz"),
        binned_expr=binned_expr[test_idx],
        expr_cont=expr_cont[test_idx],
        non_zero_mask=non_zero_mask[test_idx],
        perturbations=perturbations[test_idx]
    )
    logging.info(f"Saved test data to {os.path.join(args.output_dir, 'test_data.npz')}")

    best_corr = -float('inf')
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        epoch_start_time = time.time()
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            binned_expr = batch['binned_expr'].cuda()
            expr_cont = batch['expr_cont'].cuda()
            non_zero_mask = batch['non_zero_mask'].cuda()
            perturb_idx = batch['perturb_idx'].cuda()

            optimizer.zero_grad(set_to_none=True)
            
            with autocast():
                cont_pred = model(binned_expr, perturb_idx, non_zero_mask)
                cont_loss = torch.nn.functional.mse_loss(cont_pred, expr_cont, reduction='none')
                cont_loss = (cont_loss * non_zero_mask).sum() / (non_zero_mask.sum() + 1e-8)

            scaler.scale(cont_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += cont_loss.item()

        avg_loss = total_loss / len(train_loader)
        epoch_time = time.time() - epoch_start_time
        logging.info(f"Epoch {epoch+1}: Avg MSE Loss = {avg_loss:.4f}, Time: {epoch_time:.2f}s")

        with torch.no_grad():
            logging.info(f"[Eval @ Epoch {epoch+1}] -----------------------------")
            corr = evaluate(model, test_loader)
            logging.info(f"Epoch {epoch+1}: Mean Pearson Correlation = {corr:.4f}")
            
            if corr > best_corr:
                best_corr = corr
                best_path = os.path.join(args.output_dir, f"best_model.pt")
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'correlation': corr,
                    'loss': avg_loss,
                }, best_path)
                logging.info(f"New best model saved with correlation {corr:.4f}")

        ckpt_path = os.path.join(args.output_dir, f"checkpoint_epoch{epoch+1}.pt")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'correlation': corr,
            'loss': avg_loss,
        }, ckpt_path)


def main():
    """Parse arguments and start training."""
    parser = argparse.ArgumentParser(description="Train BioFormer perturbation model")
    parser.add_argument('--h5ad', type=str, required=True,
                        help='Path to h5ad file containing perturbation data')
    parser.add_argument('--hvg_file', type=str, required=True,
                        help='Path to file with selected genes (one per line)')
    parser.add_argument('--perturb_col', type=str, required=True,
                        help='Column in adata.obs containing perturbation labels')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save model checkpoints and logs')
    parser.add_argument('--pretrained_ckpt', type=str, default=None,
                        help='Path to pretrained model checkpoint')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs to train')
    parser.add_argument('--num_bins', type=int, default=51,
                        help='Number of bins for expression quantization')
    parser.add_argument('--d_model', type=int, default=512,
                        help='Model dimension')
    parser.add_argument('--nhead', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=8,
                        help='Number of transformer layers')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()