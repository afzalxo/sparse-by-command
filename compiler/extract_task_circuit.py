import argparse
import torch
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import os

from models.carla_model import CarlaVLAHWOld as CarlaVLAHW
from models.carla_model import MaskedCarlaVLA
from dataloaders.carla_dataloaders import CarlaImageCache, CarlaFastDataset, collate_fn
from dataloaders.carla_augmentations import GPUAugmenter
from loss.branched_loss import branched_loss
from carla_utils.eval_utils import evaluate_in_carla
from carla_utils.carla_server import CarlaServer
import random
import time
import carla

CSV_PATH = "carla_ue4_dataset_base/data.csv"
IMG_DIR = "carla_ue4_dataset_base/images"
IMG_SIZE = 256
STACK_FRAMES = 1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def extract_circuit():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default="follow lane", help="e.g., 'follow lane', 'turn left'")
    parser.add_argument('--model_path', type=str, default="carla_vla_best_driver.pth")
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lambda_sparse', type=float, default=0.01, help="Higher = Sparser")
    args = parser.parse_args()

    print(f"=== Extracting N-ISA Circuit for Task: '{args.task}' ===")

    # 1. LOAD AND FILTER DATASET
    full_df = pd.read_csv(CSV_PATH)
    # Filter strictly for the requested task
    task_df = full_df[full_df['instruction'].str.lower() == args.task.lower()].reset_index(drop=True)
    task_df = task_df[task_df['is_train_candidate'] == 1].reset_index(drop=True)
    task_df['image_path'] = IMG_DIR + '/' + task_df['image_path']
    
    if len(task_df) == 0:
        print("Error: No data found for that task!")
        return

    print(f"Found {len(task_df)} samples for '{args.task}'.")
    # task_df = task_df.iloc[:2048, :]
    
    # Setup DataLoaders
    train_indices = task_df.index.tolist()
    # (Assuming CarlaFastDataset accepts DataFrame directly in your current setup, 
    # adjust if using CacheManager)
    cache_manager = CarlaImageCache(task_df, IMG_SIZE)
    train_ds = CarlaFastDataset(cache_manager, train_indices, flip_prob=0.0, is_train=False) # Turn off random flips
    
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, 
                              num_workers=4, pin_memory=True, collate_fn=collate_fn)

    # 2. PREPARE MASKED MODEL
    base_model = CarlaVLAHW(STACK_FRAMES).to(DEVICE)
    base_model.load_state_dict(torch.load(args.model_path, map_location=DEVICE), strict=True)
    
    model = MaskedCarlaVLA(base_model, oc_par=16).to(DEVICE)
    augmenter = GPUAugmenter().to(DEVICE)
    
    # Optimizer ONLY updates the masks, not the backbone weights
    optimizer = optim.Adam(model.masks.parameters(), lr=0.05)

    # 3. OPTIMIZE MASKS
    for epoch in range(args.epochs):
        total_loss, total_drive, total_sparse = 0, 0, 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for imgs, branch_ids, prevs, targets in loop:
            imgs = imgs.to(DEVICE, non_blocking=True)
            branch_ids = branch_ids.to(DEVICE, non_blocking=True)
            prevs = prevs.to(DEVICE, non_blocking=True)
            targets = targets.to(DEVICE, non_blocking=True)
            
            imgs = augmenter(imgs, is_train=False)
            
            optimizer.zero_grad()
            
            # Forward pass through masks
            preds = model(imgs, prevs)
            
            # Calculate Losses
            driving_loss, _ = branched_loss(preds, targets, branch_ids)
            sparsity_loss = model.get_sparsity_loss()
            
            # The Tug-of-War
            loss = driving_loss + (args.lambda_sparse * sparsity_loss)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_drive += driving_loss.item()
            total_sparse += sparsity_loss.item()
            
            loop.set_postfix(
                DriveL=f"{driving_loss.item():.3f}", 
                SparseL=f"{sparsity_loss.item():.1f}"
            )

    # 4. EXTRACT AND SAVE COMPILED PACKET
    final_masks = model.get_binary_masks(threshold=0.5)

    model = base_model.to(DEVICE)
    
    total_tiles = 0
    active_tiles = 0
    print("\n--- Final N-ISA Tile Configuration ---")
    for layer, mask in final_masks.items():
        t = len(mask)
        a = sum(mask)
        total_tiles += t
        active_tiles += a
        print(f"{layer:<8} | Active: {a:>2}/{t:<2} | Mask: {''.join(map(str, mask))}")
        
    print(f"\nOverall Sparsity: {100 * (1 - active_tiles/total_tiles):.1f}%")
    
    model.eval()
  
    with torch.no_grad():
        conv_layers = model.get_conv_layers()
        for i, layer in enumerate(conv_layers):
            mask = final_masks[f'conv{i+1}']
            for tile_idx, bit in enumerate(mask):
                if bit == 0:
                    start_ch = tile_idx * 16
                    end_ch = start_ch + 16
                    layer.weight[start_ch:end_ch, :, :, :] = 0.0
                    
                    if layer.bias is not None:
                        layer.bias[start_ch:end_ch] = 0.0

    total_route_covered = 0.0
    total_route_length = 0.0
    total_cte = 0.0
    total_frame_count = 0
    route_results = {}

    client_port = random.randint(30000, 40000)
    carla_server = CarlaServer(rpc_port=client_port)
    carla_proc = carla_server.start()
    client = carla.Client("localhost", client_port)
    client.set_timeout(60.0)
    client.load_world("Town10HD_Opt")
    time.sleep(5)
    world = client.get_world()
    old_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    settings.substepping = True
    settings.max_substep_delta_time = 0.01
    settings.max_substeps = 10
    world.apply_settings(settings)
    weather = carla.WeatherParameters.ClearNoon
    world.set_weather(weather)

    ROUTES_TO_EVAL = [(66, 87)]  # Follow lane only
    ROUTES_EVAL_NAMES = {
        (66, 87): "Follow Lane",
    }

    eval_scenario = "base"
    route_indices = ROUTES_TO_EVAL[0]
    EVAL_RENDER_SIZE = (1920, 1080)
    FOV = 110
    USE_AUX_CAMERA_FOR_EVAL = False
    AUX_CAMERA_SIZE = (128, 128)
    route_covered, route_length, cte, frame_count = evaluate_in_carla(model,
                                        0, 
                                        DEVICE, 
                                        client_port=client_port,
                                        world=world,
                                        eval_scenario=eval_scenario,
                                        n_vehicles=0,
                                        n_walkers=0,
                                        spawn_index=route_indices[0],
                                        dest_index=route_indices[1],
                                        eval_render_size=EVAL_RENDER_SIZE,
                                        fov=FOV,
                                        use_aux_camera=USE_AUX_CAMERA_FOR_EVAL,
                                        aux_camera_size=AUX_CAMERA_SIZE,
                                        save_to_wandb=False)
    route_name = ROUTES_EVAL_NAMES.get(route_indices, f"{route_indices[0]}->{route_indices[1]}")
    route_results[f"Completion {route_name} {route_indices[0]}->{route_indices[1]}"] = route_covered / route_length * 100.0 if route_length > 0 else 0.0
    route_results[f"CTE {route_name} {route_indices[0]}->{route_indices[1]}"] = cte / frame_count if frame_count > 0 else 0.0
    total_route_covered += route_covered
    total_route_length += route_length
    total_cte += cte
    total_frame_count += frame_count
    completion = (total_route_covered / total_route_length) * 100.0 if total_route_length > 0 else 0.0
    cte = total_cte / total_frame_count if total_frame_count > 0 else 0.0

    # Save to JSON
    output_filename = f"circuit_{args.task.replace(' ', '_')}.json"
    with open(output_filename, "w") as f:
        json.dump(final_masks, f, indent=4)
    print(f"Saved N-ISA Configuration Packet to {output_filename}")
    carla_server.stop()

if __name__ == "__main__":
    extract_circuit()
