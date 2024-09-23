import torch
import torch.nn as nn
import torch.nn.functional as F


def inject_trainable_lora(model, target_replace_module, r=4, loras=None, dropout_p=0.0):
    require_grad_params = []
    names = []

    for name, module in model.named_modules():
        if any(t in str(type(module)) for t in target_replace_module):
            lora_module = LoRALayer(module, r, dropout_p)
            module.forward = lora_module.forward
            require_grad_params.append(lora_module.parameters())
            names.append(name)

    return require_grad_params, names

class LoRALayer(nn.Module):
    def __init__(self, module, r, dropout_p):
        super().__init__()
        self.module = module
        self.lora_down = nn.Linear(module.in_features, r, bias=False)
        self.lora_up = nn.Linear(r, module.out_features, bias=False)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        return self.module(x) + self.lora_up(self.dropout(self.lora_down(x)))