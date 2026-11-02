import torch, torch.nn as nn, torch.nn.functional as F
from models.carla_model import TileGatingNetwork
OC = 16

class PrunedGatedCNN(nn.Module):
    def __init__(self, surv_counts, hidden_dim=512, task_dim=6):
        super().__init__()
        self.oc_par = OC
        ch = [3] + [c * OC for c in surv_counts]
        self.conv_layers = nn.ModuleList([nn.Conv2d(ch[i], ch[i+1], 3, padding=1, stride=2) for i in range(8)])
        self.total_tiles_per_layer = list(surv_counts)
        self.predictable_tiles = [c - 1 for c in surv_counts]
        self.vision_dim = ch[-1]
        self.measure_proj = nn.Linear(3, 128)
        self.fusion = nn.Linear(self.vision_dim + 128, hidden_dim)
        self.dropout = nn.Dropout(0.5)
        self.branches = nn.ModuleList([nn.Sequential(
            nn.Linear(hidden_dim, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 2), nn.Tanh()) for _ in range(6)])
        self.gater = TileGatingNetwork(task_dim, self.predictable_tiles)

    def get_conv_layers(self):
        return list(self.conv_layers)

    def apply_mask(self, x, pm, hard=False):
        B, C, H, W = x.shape
        nt = C // self.oc_par
        if hard:
            b = (pm > 0.5).float(); pm = (b - pm).detach() + pm
        full_m = torch.cat([torch.ones(B, 1, device=x.device), pm], 1)
        return (x.view(B, nt, self.oc_par, H, W) * full_m.view(B, nt, 1, 1, 1)).view(B, C, H, W)

    def forward(self, img, prev, task_vec, hard_masking=False, dense=False):
        masks = self.gater(task_vec)
        sp = torch.split(masks, self.predictable_tiles, 1)
        x = img
        for i, c in enumerate(self.conv_layers):
            x = c(x)
            if not dense:
                x = self.apply_mask(x, sp[i], hard_masking)
            x = F.relu(x)
        vis = x.view(x.size(0), -1)
        emb = F.relu(self.fusion(torch.cat([vis, F.relu(self.measure_proj(prev))], 1)))
        emb = self.dropout(emb)
        return torch.stack([b(emb) for b in self.branches], 1), masks



def load_pruned(cfg_path, ckpt_path=None, device="cpu"):
    import json
    surv = json.load(open(cfg_path))["surv_counts"]
    m = PrunedGatedCNN(surv).to(device)
    if ckpt_path:
        sd = torch.load(ckpt_path, map_location=device)
        m.load_state_dict(sd["state"] if "state" in sd else sd)
    return m, surv
