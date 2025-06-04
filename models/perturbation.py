"""
BioFormerPerturb model for gene expression perturbation prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BioFormerPerturb(nn.Module):
    """
    BioFormer model for predicting gene expression after perturbation.
    Uses transformer architecture with gene, value, and perturbation embeddings.
    """
    
    def __init__(self, vocab_size, num_perturbs, num_bins=51, d_model=512, 
                 nhead=8, num_layers=8, dropout=0.1):
        """
        Initialize BioFormerPerturb model.
        
        Parameters:
        -----------
        vocab_size : int
            Number of genes in the vocabulary
        num_perturbs : int
            Number of unique perturbation types
        num_bins : int, default=51
            Number of expression value bins (including zero)
        d_model : int, default=512
            Model dimension for embeddings and transformer
        nhead : int, default=8
            Number of attention heads
        num_layers : int, default=8
            Number of transformer encoder layers
        dropout : float, default=0.1
            Dropout rate for transformer layers
        """
        super().__init__()
        self.d_model = d_model
        self.num_bins = num_bins

        self.gene_embedding = nn.Embedding(vocab_size, d_model)
        self.value_embedding = nn.Embedding(num_bins, d_model)
        self.perturb_embedding = nn.Embedding(num_perturbs, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=d_model*4, 
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        self.cont_head = nn.Linear(d_model, 1)
        
        self.norm = nn.LayerNorm(d_model)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for emb in [self.gene_embedding, self.value_embedding, self.perturb_embedding]:
            nn.init.xavier_uniform_(emb.weight)
            
        for p in self.transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
        nn.init.xavier_uniform_(self.cont_head.weight)

    def forward(self, binned_expr, perturb_idx, non_zero_mask=None):
        """
        Forward pass for the model.
        
        Parameters:
        -----------
        binned_expr : torch.Tensor [batch_size, seq_len]
            Binned expression values
        perturb_idx : torch.Tensor [batch_size]
            Perturbation indices
        non_zero_mask : torch.Tensor [batch_size, seq_len], optional
            Mask for non-zero expression values
            
        Returns:
        --------
        torch.Tensor [batch_size, seq_len]
            Predicted continuous expression values
        """
        batch_size, seq_len = binned_expr.shape
        device = binned_expr.device
        
        gene_ids = torch.arange(seq_len, device=device).expand(batch_size, seq_len)

        emb = (self.gene_embedding(gene_ids) + 
               self.value_embedding(binned_expr) + 
               self.perturb_embedding(perturb_idx).unsqueeze(1))

        emb = self.norm(emb)
        
        if non_zero_mask is not None:
            emb = emb * non_zero_mask.unsqueeze(-1)

        x = self.transformer(emb)
        
        cont_pred = self.cont_head(x).squeeze(-1)
        return cont_pred