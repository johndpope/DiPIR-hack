import torch
import torch.nn.functional as F

def consistency_loss(env_fg, env_shadow):
    L_fg = torch.sum(env_fg(torch.ones_like(env_fg.mu)), dim=-1)
    L_shadow = torch.sum(env_shadow(torch.ones_like(env_shadow.mu)), dim=-1)
    
    L_fg_norm = L_fg / L_fg.sum()
    L_shadow_norm = L_shadow / L_shadow.sum()
    
    loss = F.kl_div(L_shadow_norm.log(), L_fg_norm, reduction='batchmean')
    
    return loss

def regularization_loss(env_shadow):
    L_shadow = torch.sum(env_shadow(torch.ones_like(env_shadow.mu)), dim=-1)
    loss = torch.log(1 + 2 * (L_shadow ** 2)).mean()
    return loss

def fuse_environment_maps(env_fg, env_shadow, ratio):
    L_fg = torch.sum(env_fg(torch.ones_like(env_fg.mu)), dim=-1)
    L_shadow = torch.sum(env_shadow(torch.ones_like(env_shadow.mu)), dim=-1)
    
    r = (L_fg / L_fg.max()) * (L_shadow / (L_fg + L_shadow))
    
    L_fused = (1 - r) * L_fg + r * L_shadow
    
    scale_factor = L_fused / L_fg
    env_fg.c.data *= scale_factor.unsqueeze(-1)
    
    env_fg.c.data = ratio * env_fg.c.data + (1 - ratio) * env_shadow.c.data
    env_shadow.c.data = env_fg.c.data