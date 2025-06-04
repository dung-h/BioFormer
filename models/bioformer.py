import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class GradientReversal(torch.autograd.Function):
    """Gradient Reversal Layer for adversarial training"""
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class BioFormer(nn.Module):
    """BioFormer model architecture"""
    def __init__(self, vocab_size, num_cell_types, num_studies=None, num_bins=51, 
                d_model=512, nhead=8, num_layers=12, dropout=0.1):
        super(BioFormer, self).__init__()
        self.d_model = d_model
        self.num_bins = num_bins
        
        self.gene_embedding = nn.Embedding(vocab_size, d_model)
        self.value_embedding = nn.Embedding(num_bins, d_model)
        self.cell_type_embedding = nn.Embedding(num_cell_types, d_model)
        
        if num_studies is not None:
            self.study_embedding = nn.Embedding(num_studies, d_model)
        
        encoder_layer = TransformerEncoderLayer(d_model, nhead, 
                                               dim_feedforward=d_model*4, 
                                               dropout=dropout, 
                                               batch_first=True)
        self.transformer = TransformerEncoder(encoder_layer, num_layers)
        
        self.mlm_head = nn.Linear(d_model, num_bins)  # MLM
        self.cont_head = nn.Linear(d_model, 1)        # Continuous expression
        
        if num_studies is not None:
            self.adv_head = nn.Linear(d_model, num_studies)
        
        self.norm = nn.LayerNorm(d_model)
        
        self._init_weights()
    
    def _init_weights(self):
        embeddings = [self.gene_embedding, self.value_embedding, self.cell_type_embedding]
        if hasattr(self, 'study_embedding'):
            embeddings.append(self.study_embedding)
            
        for emb in embeddings:
            nn.init.xavier_uniform_(emb.weight)
        
        for p in self.transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
        nn.init.xavier_uniform_(self.mlm_head.weight)
        nn.init.xavier_uniform_(self.cont_head.weight)
        
        if hasattr(self, 'adv_head'):
            nn.init.xavier_uniform_(self.adv_head.weight)

    def forward(self, binned_expr, cell_type, study_id=None, non_zero_mask=None, 
               adv_alpha=1.0, return_attention=False):
        batch_size, seq_len = binned_expr.shape
        device = binned_expr.device

        gene_emb = self.gene_embedding(torch.arange(seq_len, device=device).expand(batch_size, seq_len))
        value_emb = self.value_embedding(binned_expr)
        cell_type_emb = self.cell_type_embedding(cell_type).unsqueeze(1)
        
        if hasattr(self, 'study_embedding') and study_id is not None:
            study_emb = self.study_embedding(study_id).unsqueeze(1)
            emb = gene_emb + value_emb + cell_type_emb + study_emb
        else:
            emb = gene_emb + value_emb + cell_type_emb
            
        emb = self.norm(emb)

        if non_zero_mask is not None:
            emb = emb * non_zero_mask.unsqueeze(-1)

        if return_attention:
            attention_outputs = []

            def save_attention(module, input, output):
                attention_outputs.append(output[1])

            handles = []
            for layer in self.transformer.layers:
                handle = layer.self_attn.register_forward_hook(save_attention)
                handles.append(handle)

            output = self.transformer(emb)

            for h in handles:
                h.remove()

            mlm_logits = self.mlm_head(output)
            cont_pred = self.cont_head(output).squeeze(-1)
            
            adv_pred = None
            if hasattr(self, 'adv_head') and study_id is not None:
                cls_embeddings = output[:, 0]  # Use CLS token embedding
                rev_features = GradientReversal.apply(cls_embeddings, adv_alpha)
                adv_pred = self.adv_head(rev_features)
            
            return mlm_logits, cont_pred, output, attention_outputs, adv_pred
        else:
            output = self.transformer(emb)
            mlm_logits = self.mlm_head(output)
            cont_pred = self.cont_head(output).squeeze(-1)
            
            adv_pred = None
            if hasattr(self, 'adv_head') and study_id is not None:
                cls_embeddings = output[:, 0]  # Use CLS token embedding
                rev_features = GradientReversal.apply(cls_embeddings, adv_alpha)
                adv_pred = self.adv_head(rev_features)
            
            return mlm_logits, cont_pred, output, adv_pred