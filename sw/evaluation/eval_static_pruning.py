import torch, sys, os, json, time
import multiprocessing
sys.path.append("/path/to/vla-carla")
from models.carla_model import CarlaVLAHW_Gated
from carla_utils.eval_utils import evaluate_single_route_wrapper

MODEL_PATH = "/path/to/vla-carla/model_checkpoints_sparse/carla_vla_furthest_driver_phase3_gated.pth"
EVAL_SCENARIOS = ["dynamic"]
ROUTES = [(31, 99), (13, 34), (72, 106)]
ROUTE_NAMES = {(31, 99): "Hard 1", (13, 34): "Hard 2", (72, 106): "Hard 3"}
EVAL_RENDER_SIZE = (1920, 1080)
FOV = 110
eval_gpus = [0, 3, 4]

with open("/path/to/vla-carla/static_pruning_masks.json") as f:
    all_masks = json.load(f)

OC_PAR = 16
tiles_per_layer = [2, 4, 8, 16, 32, 32, 32, 64]

# For each sparsity level, we need to override the gating network output
# with the static mask. We do this by modifying the model to ignore the 
# gating network and use fixed masks.

sparsity_levels = ["0", "20", "40", "50", "60", "62", "70", "80"]

for sp_key in sparsity_levels:
    info = all_masks[sp_key]
    masks_per_layer = info["masks"]
    actual_active = info["actual_active"]
    actual_sparsity = info["actual_sparsity"]
    
    print("\n" + "="*60)
    print("EVALUATING SPARSITY %s%% (%d/190 active, %.1f%% pruned)" % (
        sp_key, actual_active, actual_sparsity*100))
    print("="*60)
    
    # Convert masks to the format the model expects
    # The gating network outputs 182 values (excluding tile 0 per layer)
    # We need to set these to match our static mask
    gated_tiles = [t - 1 for t in tiles_per_layer]
    static_gater_output = []
    for l, n_tiles in enumerate(tiles_per_layer):
        mask_val = masks_per_layer[l]
        for t in range(1, n_tiles):  # Skip tile 0 (always on)
            bit = (mask_val >> t) & 1
            static_gater_output.append(float(bit))
    
    static_mask_tensor = torch.tensor([static_gater_output], dtype=torch.float32)
    
    # Save a temporary model with overridden gater
    model = CarlaVLAHW_Gated(task_dim=6)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    
    # Override the gater to always return the static mask
    class StaticGater(torch.nn.Module):
        def __init__(self, mask):
            super().__init__()
            self.register_buffer("mask", mask)
        def forward(self, task_vec):
            return self.mask.expand(task_vec.shape[0], -1)
    
    model.gater = StaticGater(static_mask_tensor)
    
    tmp_path = "/path/to/vla-carla/tmp_static_%s.pth" % sp_key
    torch.save(model.state_dict(), tmp_path)
    
    # Run eval on 3 routes in parallel
    ctx = multiprocessing.get_context("spawn")
    queues = []
    processes = []
    for i, route_indices in enumerate(ROUTES):
        gpu_id = eval_gpus[i % len(eval_gpus)]
        route_name = ROUTE_NAMES.get(route_indices, "%d->%d" % route_indices)
        q = ctx.Queue()
        queues.append(q)
        p = ctx.Process(target=evaluate_single_route_wrapper,
                        args=(tmp_path, 0, torch.device("cuda:%d" % gpu_id),
                              EVAL_SCENARIOS, route_indices, route_name,
                              EVAL_RENDER_SIZE, FOV, True, False, gpu_id, True, q))
        processes.append(p)
    
    for p in processes:
        p.start()
        time.sleep(3)
    for p in processes:
        p.join(timeout=300)
    
    total_rc, total_rl, total_cte, total_fc = 0.0, 0.0, 0.0, 0
    for q in queues:
        try:
            rc, rl, cte_sum, fc, rr = q.get(timeout=10)
            total_rc += rc
            total_rl += rl
            total_cte += cte_sum
            total_fc += fc
        except:
            total_rl = max(total_rl, 1.0)
            total_fc = max(total_fc, 1)
    
    completion = (total_rc / total_rl) * 100.0 if total_rl > 0 else 0.0
    cte = total_cte / total_fc if total_fc > 0 else 999.0
    
    print("RESULT sparsity=%s active=%d completion=%.1f cte=%.3f" % (
        sp_key, actual_active, completion, cte))
    
    # Clean up
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

print("\nDone!")
