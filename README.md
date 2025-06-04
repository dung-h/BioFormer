# BioFormer

**BioFormer** is a transformer-based deep learning model for single-cell RNA-seq representation learning. It is trained on large-scale CELLxGENE datasets and supports downstream applications like clustering and perturbation prediction.

## 🧠 Features

- Transformer encoder architecture with no positional embeddings  
- Multi-task objectives: MLM, continuous expression, adversarial loss, and ECS loss  
- Compatible with distributed (DDP) and mixed precision training (AMP)  
- Modular scripts for pretraining, perturbation fine-tuning, and UMAP evaluation

## 🚀 Setup

```bash
git clone https://github.com/yourname/BioFormer.git
cd BioFormer
pip install -r requirements.txt
```

## 📊 Pretraining on CELLxGENE

```bash
python3 -m scripts.train \
  --data_dir /nfsshared/preprocessed \
  --output_dir /nfsshared/training \
  --batch_size 64 \
  --epochs 10 \
  --lr 1e-4
```

## 🧪 Fine-tuning for Perturbation Prediction

```bash
python3 -m scripts.train_perturbation \
  --h5ad /home/tripham/scgpt/test_clustering/perturb/data.h5ad \
  --hvg_file /nfsshared/preprocessed/selected_genes.txt \
  --perturb_col guide_identity \
  --output_dir /nfsshared/training \
  --pretrained_ckpt /nfsshared/training/checkpoints/checkpoint_epoch_1_20250602_213400.pt
```

## 📈 Evaluation on Clustering Task

```bash
python3 -m scripts.evaluate_clustering \
  --data_dir /home/tripham/scgpt/test_clustering/test3 \
  --selected_genes_file /nfsshared/preprocessed/selected_genes.txt \
  --checkpoint_path /nfsshared/training/checkpoints/checkpoint_epoch_1_20250602_213400.pt \
  --output_dir results/

```
