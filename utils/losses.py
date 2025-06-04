import torch
import torch.nn.functional as F

def ecs_loss(embeddings, cell_type, margin=1.0):
    embeddings = embeddings[:, 0]  # Use CLS token
    dist_matrix = torch.cdist(embeddings, embeddings)
    same_type = (cell_type.unsqueeze(1) == cell_type.unsqueeze(0)).float()
    loss = (same_type * dist_matrix.pow(2) + 
            (1 - same_type) * F.relu(margin - dist_matrix).pow(2)).mean()
    return loss

def bio_consistency_loss(embeddings, cell_type):
    
    embeddings = embeddings[:, 0]  # Use CLS token
    dist_matrix = torch.cdist(embeddings, embeddings)
    same_type = (cell_type.unsqueeze(1) == cell_type.unsqueeze(0)).float()
    loss = (same_type * dist_matrix.pow(2)).mean()
    return loss

def contrastive_loss(embeddings, cell_type, study_id, temperature=0.5):
    
    embeddings = embeddings[:, 0]  # Use CLS token
    embeddings = F.normalize(embeddings, dim=1)
    sim_matrix = torch.mm(embeddings, embeddings.t()) / temperature
    exp_sim = torch.exp(sim_matrix)
    
    same_cell_type = (cell_type.unsqueeze(1) == cell_type.unsqueeze(0)).float()
    different_study = (study_id.unsqueeze(1) != study_id.unsqueeze(0)).float()
    pos_mask = same_cell_type * different_study
    
    pos_sim = (exp_sim * pos_mask).sum(dim=1)
    total_sim = exp_sim.sum(dim=1)
    loss = -torch.log(pos_sim / total_sim + 1e-8).mean()
    return loss