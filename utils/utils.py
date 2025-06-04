"""
General utility functions for BioFormer.
"""

import torch


def load_model_with_mismatch(model, checkpoint_path):
    """
    Load model from checkpoint handling parameter shape mismatch.
    
    Parameters:
    -----------
    model : torch.nn.Module
        The model to load weights into
    checkpoint_path : str
        Path to the checkpoint file
        
    Returns:
    --------
    None
    """
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    
    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    
    own_state = model.state_dict()
    
    for name, param in state_dict.items():
        if name in own_state:
            if own_state[name].shape == param.shape:
                own_state[name].copy_(param)
            else:
                print(f"Skipping {name}: shape mismatch {own_state[name].shape} vs {param.shape}")
        else:
            print(f"Skipping {name}: not found in current model")