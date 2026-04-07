import torch
import torch.nn as nn
import random
import numpy as np

class GPUAugmenter(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, img_stack, is_train=True):
        # Input: [B, 12, H, W] uint8 tensor
        x = img_stack.float() / 255.0
        
        b, c, h, w = x.shape
        x = x.reshape(-1, 3, h, w)
        
        if is_train:
            # Brightness
            if random.random() < 0.8:
                factor = torch.empty(x.size(0), 1, 1, 1, device=x.device).uniform_(0.7, 1.3)
                x = x * factor
            
            # Contrast
            if random.random() < 0.8:
                factor = torch.empty(x.size(0), 1, 1, 1, device=x.device).uniform_(0.7, 1.3)
                mean = x.mean(dim=(1, 2, 3), keepdim=True)
                x = (x - mean) * factor + mean

            x = torch.clamp(x, 0, 1)

        x = (x - self.mean) / self.std
        x = x.reshape(b, c, h, w)
        return x




