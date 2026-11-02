import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter

class SimpleTokenizer:
    def __init__(self, texts):
        word_counts = Counter()
        for text in texts: word_counts.update(text.lower().split())
        self.vocab = {word: i+1 for i, (word, _) in enumerate(word_counts.items())}
        self.vocab_size = len(self.vocab) + 1
        
    def encode(self, text):
        return torch.tensor([self.vocab.get(w, 0) for w in text.lower().split()], dtype=torch.long)

class CarlaVLASmall(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        
        self.measure_proj = nn.Linear(3, 64, bias=False)
        
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, stride=2)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=2)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2)

        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2)

        self.conv5 = nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=2)

        # self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.vision_dim = 128

        self.fusion = nn.Linear(self.vision_dim + 64, hidden_dim)
        self.dropout = nn.Dropout(0.5)

        self.num_branches = 1
        
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 32),
                nn.ReLU(),
                nn.Linear(32, 2),
                nn.Tanh()
            ) for _ in range(self.num_branches)
        ])


    def forward(self, img, prev_action):
        x = F.relu(self.conv1(img))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))

        # x = self.global_pool(x)
        vis_feat = x.view(x.size(0), -1)

        meas_feat = F.relu(self.measure_proj(prev_action))

        embedding = torch.cat([vis_feat, meas_feat], dim=1)
        embedding = F.relu(self.fusion(embedding))
        embedding = self.dropout(embedding)

        outputs = []
        for branch in self.branches:
            outputs.append(branch(embedding))
            
        return torch.stack(outputs, dim=1)

    def get_conv_layers(self):
        return [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5]


class TileGatingNetwork(nn.Module):
    def __init__(self, task_dim, tile_counts_to_predict):
        super().__init__()
        self.total_predictable_tiles = sum(tile_counts_to_predict)
        
        # A lightweight MLP to decode the task vector into tile masks
        self.mlp = nn.Sequential(
            nn.Linear(task_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.total_predictable_tiles)
        )
        
    def forward(self, task_vec):
        logits = self.mlp(task_vec)
        # Sigmoid forces masks to be between 0.0 (Off) and 1.0 (On)
        return torch.sigmoid(logits)


class CarlaVLAHW_Gated(nn.Module):
    def __init__(self, hidden_dim=512, task_dim=8, oc_par=16):
        super().__init__()
        self.oc_par = oc_par
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=2)
        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=2)
        self.conv7 = nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=2)
        self.conv8 = nn.Conv2d(512, 1024, kernel_size=3, padding=1, stride=2)
        # self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=1, stride=2)
        # self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=1, stride=1)
        
        # self.flatten = nn.Flatten()
        # self.flatten = nn.AdaptiveAvgPool2d((1, 1))
        self.vision_dim = 1024 # 4*4*1024 # 256 * 2 * 2
        
        self.measure_proj = nn.Linear(3, 128)
        self.fusion = nn.Linear(self.vision_dim + 128, hidden_dim)
        self.dropout0 = nn.Dropout(0.2)
        self.dropout = nn.Dropout(0.5)
        
        self.num_branches = 6
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, 256), nn.ReLU(),
                nn.Linear(256, 256), nn.ReLU(),
                nn.Linear(256, 2), nn.Tanh()
            ) for _ in range(self.num_branches)
        ])

        # --- N-ISA GATING LOGIC ---
        self.conv_layers = [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6, self.conv7, self.conv8]
        self.total_tiles_per_layer = [layer.out_channels // self.oc_par for layer in self.conv_layers]
        
        # We subtract 1 because Tile 0 is ALWAYS ON (The Polysemantic Core)
        self.predictable_tiles = [count - 1 for count in self.total_tiles_per_layer]
        self.gater = TileGatingNetwork(task_dim, self.predictable_tiles)

    def get_conv_layers(self):
        return self.conv_layers

    def apply_mask(self, x, predicted_mask, hard_masking=False):
        B, C, H, W = x.shape
        num_tiles = C // self.oc_par
        
        if hard_masking:
            # Simulate FPGA discrete execution (0.0 or 1.0)
            binary_mask = (predicted_mask > 0.5).float()
            predicted_mask = (binary_mask - predicted_mask).detach() + predicted_mask
            
        # Create the full mask:[Always ON Tile] + [Predicted Tiles]
        always_on = torch.ones(B, 1, device=x.device)
        full_mask = torch.cat([always_on, predicted_mask], dim=1) #[B, num_tiles]
        
        # Reshape for broadcasting over channels
        x_reshaped = x.view(B, num_tiles, self.oc_par, H, W)
        mask_reshaped = full_mask.view(B, num_tiles, 1, 1, 1)
        
        x_masked = x_reshaped * mask_reshaped
        return x_masked.view(B, C, H, W)

    def forward(self, img, prev_action, task_vec, hard_masking=False):
        # 1. Predict the task-specific masks
        all_pred_masks = self.gater(task_vec)
        
        # Split the flat mask vector into chunks for each layer
        split_masks = torch.split(all_pred_masks, self.predictable_tiles, dim=1)
        
        # 2. Forward Pass
        x = img
        for i, conv_layer in enumerate(self.conv_layers):
            x = conv_layer(x)
            x = self.apply_mask(x, split_masks[i], hard_masking)
            x = F.relu(x)
            
        vis_feat = x.view(x.size(0), -1)
        meas_feat = F.relu(self.measure_proj(prev_action))
        
        embedding = torch.cat([vis_feat, meas_feat], dim=1)
        embedding = F.relu(self.fusion(embedding))
        embedding = self.dropout(embedding)
        
        outputs = [branch(embedding) for branch in self.branches]
        
        # Return outputs AND the masks so we can apply sparsity loss
        return torch.stack(outputs, dim=1), all_pred_masks

    def forward_latency_bench(self, img, prev_action, split_masks, task_idx):
        x = img
        for i, conv_layer in enumerate(self.conv_layers):
            x = conv_layer(x)
            x = self.apply_mask(x, split_masks[i], hard_masking=True)
            x = F.relu(x)
            
        vis_feat = x.view(x.size(0), -1)
        meas_feat = F.relu(self.measure_proj(prev_action))
        
        embedding = torch.cat([vis_feat, meas_feat], dim=1)
        embedding = F.relu(self.fusion(embedding))
        
        # Compute output for only the relevant branch:
        outputs = self.branches[task_idx](embedding)

        return outputs


class CarlaVLAHW_Dense(nn.Module):
    def __init__(self, hidden_dim=512):
        super().__init__()
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=2)
        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=2)
        self.conv7 = nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=2)
        self.conv8 = nn.Conv2d(512, 1024, kernel_size=3, padding=1, stride=2)
        
        self.vision_dim = 1024
        
        self.measure_proj = nn.Linear(3, 128)
        self.fusion = nn.Linear(self.vision_dim + 128, hidden_dim)
        self.dropout = nn.Dropout(0.5)
        
        self.num_branches = 6
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, 256), nn.ReLU(),
                nn.Linear(256, 256), nn.ReLU(),
                nn.Linear(256, 2), nn.Tanh()
            ) for _ in range(self.num_branches)
        ])
        self.conv_layers = [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6, self.conv7, self.conv8]

    def get_conv_layers(self):
        return self.conv_layers

    def forward(self, img, prev_action):
        # 2. Forward Pass
        x = img
        for i, conv_layer in enumerate(self.conv_layers):
            x = conv_layer(x)
            x = F.relu(x)
            
        vis_feat = x.view(x.size(0), -1)
        meas_feat = F.relu(self.measure_proj(prev_action))
        
        embedding = torch.cat([vis_feat, meas_feat], dim=1)
        embedding = F.relu(self.fusion(embedding))
        embedding = self.dropout(embedding)
        
        outputs = [branch(embedding) for branch in self.branches]
        
        # Return outputs AND the masks so we can apply sparsity loss
        return torch.stack(outputs, dim=1)

