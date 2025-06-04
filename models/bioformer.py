import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class BioFormer(nn.Module):
    """BioFormer model identical to scGPT evaluation-time version (no study_id, no adv_head)."""
    def __init__(self, vocab_size, num_cell_types, num_bins=51, 
                 d_model=512, nhead=8, num_layers=12, dropout=0.1):
        super(BioFormer, self).__init__()
        self.d_model = d_model
        self.num_bins = num_bins

        # Embedding layers
        self.gene_embedding = nn.Embedding(vocab_size, d_model)
        self.value_embedding = nn.Embedding(num_bins, d_model)
        self.cell_type_embedding = nn.Embedding(num_cell_types, d_model)

        # Transformer encoder
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=4 * d_model, dropout=dropout,
            batch_first=True
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers)

        # Output heads
        self.mlm_head = nn.Linear(d_model, num_bins)
        self.cont_head = nn.Linear(d_model, 1)

        # Normalization
        self.norm = nn.LayerNorm(d_model)

        self._init_weights()

    def _init_weights(self):
        for emb in [self.gene_embedding, self.value_embedding, self.cell_type_embedding]:
            nn.init.xavier_uniform_(emb.weight)
        for p in self.transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        nn.init.xavier_uniform_(self.mlm_head.weight)
        nn.init.xavier_uniform_(self.cont_head.weight)

    def forward(self, binned_expr, cell_type, non_zero_mask=None, return_attention=False):
        batch_size, seq_len = binned_expr.size()
        device = binned_expr.device

        gene_emb = self.gene_embedding(torch.arange(seq_len, device=device).expand(batch_size, seq_len))
        value_emb = self.value_embedding(binned_expr)
        cell_type_emb = self.cell_type_embedding(cell_type).unsqueeze(1)

        emb = gene_emb + value_emb + cell_type_emb
        emb = self.norm(emb)

        if non_zero_mask is not None:
            emb = emb * non_zero_mask.unsqueeze(-1)

        if return_attention:
            attention_outputs = []

            def save_attention(module, input, output):
                attention_outputs.append(output[1])  # Get attention weights

            handles = [layer.self_attn.register_forward_hook(save_attention) for layer in self.transformer.layers]
            output = self.transformer(emb)
            for h in handles:
                h.remove()

            mlm_logits = self.mlm_head(output)
            cont_pred = self.cont_head(output).squeeze(-1)
            return mlm_logits, cont_pred, output, attention_outputs
        else:
            output = self.transformer(emb)
            mlm_logits = self.mlm_head(output)
            cont_pred = self.cont_head(output).squeeze(-1)
            return mlm_logits, cont_pred, output
