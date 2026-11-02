"""Reproduce Table 2 'Active Tiles' column from the trained gating checkpoint.
Loads the gated CNN, runs the gating MLP on each one-hot command, thresholds at 0.5,
and reports the number of active tiles (of the 182 prunable conv tiles) per command."""
import torch, sys
sys.path.insert(0, ".")
from models.carla_model import CarlaVLAHW_Gated
CKPT = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/carla_vla_furthest_driver_phase3_gated.pth"
CMD = ["Follow Lane","Turn Left","Turn Right","Ch. Lane Left","Ch. Lane Right","Brake"]
m = CarlaVLAHW_Gated(task_dim=6).eval()
m.load_state_dict(torch.load(CKPT, map_location="cpu"))
pred = m.gater(torch.eye(6))                       # [6, 182] predictable-tile logits/probs
active = (pred > 0.5).sum(1)                        # active prunable tiles per command
T = sum(m.predictable_tiles)                        # 182
print(f"checkpoint: {CKPT}")
print(f"prunable tiles: {T}\n{'Command':<15}{'Active/182':>12}{'%':>8}")
for c in range(6):
    a = int(active[c]); print(f"{CMD[c]:<15}{a:>7}/{T:<4}{100*a/T:>8.1f}")
