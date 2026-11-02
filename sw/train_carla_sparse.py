import sys
import os
import carla
import queue
import pandas as pd
import subprocess
import signal
import time
import socket
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from collections import Counter
from collections import deque
import multiprocessing
from multiprocessing import Process
import random
import gc
import cv2
import math

from models.carla_model import CarlaVLAHW_Gated as CarlaVLA
from dataloaders.carla_dataloaders import CarlaImageCache, CarlaFastDatasetGated, collate_fn_gated
from dataloaders.carla_augmentations import GPUAugmenter
from carla_utils.eval_utils import evaluate_in_carla_gated, evaluate_in_carla_mp_wrapper
from carla_utils.carla_server import CarlaServer
from loss.branched_loss import branched_loss
from sparse_extract.tile_mask_extractor import extract_tile_masks

sys.path.append("/home/eeafzal/carla_simulator/PythonAPI/carla")
from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.local_planner import RoadOption

BASE_DATASET_DIR = "carla_ue4_dataset_base"
WEATHER_DATASET_DIR = "carla_ue4_dataset_weather"
DYNAMIC_DATASET_DIR = "carla_ue4_dataset_dynamic"

EVAL_SCENARIOS = ["dynamic"]
N_VEHICLES = 40
N_WALKERS = 40

LAMBDA_SPARSE = 0.0005
TASK_DIM = 6

BATCH_SIZE = 256 
NUM_WORKERS = 8
PREFETCH_FACTOR = 2
N_TRAIN_SAMPLES = 0 # 420000
FLIP_PROB = 0.5

LEARNING_RATE = 2.0e-4
EPOCHS = 300
BEGIN_EVAL_EPOCH = 70

WEIGHT_DECAY = 1e-3
ACTION_DROPOUT_PROB = 0.50

EVAL_RENDER_SIZE = (1920, 1080)
FOV = 110
IMG_SIZE = 256

USE_AUX_CAMERA_FOR_EVAL = False
AUX_CAMERA_SIZE = (1280, 720)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval-only', action='store_true', help='Run evaluation only')
    parser.add_argument('--finetune', action='store_true', help='Finetune from a pretrained checkpoint')
    parser.add_argument('--model-path', type=str, required=False, help='Path to the model checkpoint for evaluation')
    parser.add_argument('--fast-dataset', action='store_true', help='Use fast in-RAM dataset (requires more RAM)')
    parser.add_argument('--use-wandb', action='store_true', help='Use Weights & Biases for logging')
    parser.add_argument('--no-eval', action='store_true', help='Disable evaluation during training')
    parser.add_argument('--resume', action='store_true', help='Resume training from the latest checkpoint')
    args = parser.parse_args()

    eval_only = args.eval_only
    model_path = args.model_path
    fast_dataset = args.fast_dataset
    use_wandb = args.use_wandb
    if use_wandb:
        import wandb
        wandb.init(project="vla-accel", name=f"train_gated_lambda_{LAMBDA_SPARSE}_dynamic_only")

    # 1. Load Metadata Only (Fast)
    def process_df(path, scenario_name, is_rainy, is_crowded):
        df = pd.read_csv(os.path.join(path, "data.csv"))
        # Prepend the folder path to the image filename
        df['image_path'] = os.path.join(path, "images") + "/" + df['image_path']
        # Add metadata tags for the Gating Network
        df['is_rainy'] = is_rainy
        df['is_crowded'] = is_crowded
        # Ensure trajectory_id is unique across the whole dataset
        # (Offset each by 1,000,000 to prevent overlap)
        offset = {"base": 0, "weather": 1000000, "dynamic": 2000000}
        df['trajectory_id'] += offset[scenario_name]
        return df

    df_base = process_df(BASE_DATASET_DIR, "base", 0, 0)
    df_weather = process_df(WEATHER_DATASET_DIR, "weather", 1, 0)
    df_dynamic = process_df(DYNAMIC_DATASET_DIR, "dynamic", 0, 1)
    print(f"Base Dataset: {len(df_base[df_base['is_train_candidate'] == 1])} frames, Weather Dataset: {len(df_weather[df_weather['is_train_candidate'] == 1])} frames, Dynamic Dataset: {len(df_dynamic[df_dynamic['is_train_candidate'] == 1])} frames")
    # full_df = pd.concat([df_base, df_weather, df_dynamic], ignore_index=True)
    full_df = df_dynamic

    full_df = full_df[full_df['is_train_candidate'] == 1].reset_index(drop=True)
    # if args.finetune:
    #     full_df = full_df[full_df['instruction'] == "follow lane"].reset_index(drop=True)

    model = CarlaVLA(task_dim=TASK_DIM).to(DEVICE)
    if args.finetune:
        model.load_state_dict(torch.load(args.model_path), strict=True)
        print(f"Loaded model from {args.model_path} for finetuning.")

    ROUTES_TO_EVAL = [(81, 139), (86, 4), (66, 62)]  # Follow lane only
    ROUTES_EVAL_NAMES = {
        (81, 139): "Left Turn",
        (86, 4): "Right Turn",
        (66, 62): "Lane Following",
    }
    start_epoch = 0
    easy_eval = True
    if args.resume or eval_only:
        model.load_state_dict(torch.load(model_path), strict=True)
        print(f"Loaded model from {model_path} for {'evaluation' if eval_only else 'resuming training'}.")
        resume_epoch = int(model_path.split("_epoch_")[1].split("_")[0]) if "_epoch_" in model_path else 0
        start_epoch = resume_epoch + 1
        ROUTES_TO_EVAL = [(31, 99), (13, 34), (72, 106)]
        ROUTES_EVAL_NAMES = {
            (31, 99): "Hard 1",
            (13, 34): "Hard 2",
            (72, 106): "Hard 3",
        }
        easy_eval = False
        if args.resume:
            print(f"Resuming from epoch {start_epoch} with harder evaluation routes.")

    if eval_only:
        ctx = multiprocessing.get_context('spawn')
        result_queue = ctx.Queue()
        p = ctx.Process(target=evaluate_in_carla_mp_wrapper, args=(model_path, 0, DEVICE, EVAL_SCENARIOS, ROUTES_TO_EVAL, ROUTES_EVAL_NAMES, EVAL_RENDER_SIZE, FOV, True, False, result_queue))
        p.start()
        time.sleep(5)
        p.join()
        completion, cte, route_results = result_queue.get()
        print(f"Eval Results - Completion: {completion:.2f}%, Avg CTE: {cte:.3f}m")
        return
 
    valid_df = full_df
    if N_TRAIN_SAMPLES > 0:
        # valid_df = valid_df.iloc[120000:N_TRAIN_SAMPLES+120000, :].reset_index(drop=True)
        valid_df = valid_df.iloc[:N_TRAIN_SAMPLES, :].reset_index(drop=True)
    else:
        valid_df = valid_df.reset_index(drop=True)
    
    train_indices = valid_df.index.tolist()
    
    print(f"Train Frames: {len(train_indices)}")
    
    if fast_dataset:
        print("Using Fast In-RAM Dataset...")
        cache_manager = CarlaImageCache(valid_df, IMG_SIZE)
        train_ds = CarlaFastDatasetGated(cache_manager, train_indices, flip_prob=FLIP_PROB, is_train=True)
    else:
        print("Using Disk-Based Dataset...")
        exit(0)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
                              prefetch_factor=PREFETCH_FACTOR, collate_fn=collate_fn_gated)
    
    augmenter = GPUAugmenter().to(DEVICE) 
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    criterion = branched_loss
    scaler = GradScaler()
    
    best_val_mae = float('inf')
    best_val_loss = float('inf')
    noise_levels = torch.tensor([0.01, 0.01, 0.2]).to(DEVICE)

    best_completion = 0.0
    best_cte = float('inf')
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        total_loss = 0
        total_losses = [0, 0]
        total_sparsity_loss = 0
        if epoch < 10:
            current_dropout = (epoch / 10.0) * ACTION_DROPOUT_PROB
        else:
            current_dropout = ACTION_DROPOUT_PROB

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for imgs, branch_ids, task_vecs, prevs, targets in loop:
            imgs = imgs.to(DEVICE, non_blocking=True)
            branch_ids = branch_ids.to(DEVICE, non_blocking=True)
            task_vecs = task_vecs.to(DEVICE, non_blocking=True)
            prevs = prevs.to(DEVICE, non_blocking=True)
            targets = targets.to(DEVICE, non_blocking=True)
            
            imgs = augmenter(imgs, is_train=True)
            
            if random.random() < current_dropout:
                prevs = torch.zeros_like(prevs)
            else:
                noise = torch.randn_like(prevs) * noise_levels
                prevs = prevs + noise
            
            optimizer.zero_grad(set_to_none=True)
            
            with autocast():
                preds, masks = model(imgs, prevs, task_vecs, hard_masking=False)
                driving_loss, all_losses = criterion(preds, targets, branch_ids)
                sparsity_loss = masks.mean()
                loss = driving_loss + LAMBDA_SPARSE * sparsity_loss
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            total_sparsity_loss += (LAMBDA_SPARSE * sparsity_loss.item())
            for i in range(len(total_losses)):
                total_losses[i] += all_losses[i]
            loop.set_postfix(loss=loss.item(), driving_loss=driving_loss.item(), lasso_loss=LAMBDA_SPARSE * sparsity_loss.item(), steer_loss=all_losses[0], accel_loss=all_losses[1])
        task_vec = torch.eye(TASK_DIM, dtype=torch.float32).to(DEVICE)
        masks = model.gater(task_vec)
        binary_masks = (masks > 0.5).int().tolist()
        # print(f"FPGA Masks: {binary_masks}")
        mask_sum = sum(mask for masks in binary_masks for mask in masks)
        list_len = sum(len(masks) for masks in binary_masks)
        mask_sparsity = mask_sum / list_len
        print(f"Mask Sparsity: {mask_sparsity * 100:.1f}% active")

        steer_loss = total_losses[0] / len(train_loader)
        accel_loss = total_losses[1] / len(train_loader)
        loop.set_postfix(loss=total_loss/len(train_loader), sparsity_loss=total_sparsity_loss/len(train_loader), steer_loss=steer_loss, accel_loss=accel_loss)
        print_str = f"Epoch {epoch} | Train Loss: {total_loss/len(train_loader):.5f} | Steer Loss: {steer_loss:.5f} | Accel Loss: {accel_loss:.5f}"

        if use_wandb:
            wandb.log({
                "Train Loss": total_loss/len(train_loader),
                "Steer Loss": steer_loss,
                "Accel Loss": accel_loss,
                "Sparsity Loss": total_sparsity_loss/len(train_loader),
                "Mask Sparsity": mask_sparsity,
                "Epoch": epoch
            }, step=epoch)

        if epoch >= BEGIN_EVAL_EPOCH:
            model.eval()
          
            if not args.no_eval:
                total_route_covered = 0.0
                total_route_length = 0.0
                total_cte = 0.0
                total_frame_count = 0
                route_results = {}
                tmp_weights_path = f"tmp_eval_epoch_{epoch}.pth"
                torch.save(model.state_dict(), tmp_weights_path)

                ctx = multiprocessing.get_context('spawn')
                result_queue = ctx.Queue()
                p = ctx.Process(target=evaluate_in_carla_mp_wrapper, args=(tmp_weights_path, epoch, DEVICE, EVAL_SCENARIOS, ROUTES_TO_EVAL, ROUTES_EVAL_NAMES, EVAL_RENDER_SIZE, FOV, True, False, result_queue))
                p.start()
                time.sleep(5)  # Stagger process starts to avoid overwhelming the server
                
                p.join()
                completion, cte, route_results = result_queue.get()

                if os.path.exists(tmp_weights_path):
                    os.remove(tmp_weights_path)

                print_str += f" | Completion: {completion:.2f}% | Avg CTE: {cte:.3f}m"

                if completion >= best_completion and not easy_eval:
                    if completion == best_completion:
                        print(f">>> TIE IN COMPLETION ({completion:.1f}%) - ", end="")
                        if cte < best_cte:
                            print(f"NEW BEST CTE ({cte:.3f}m vs {best_cte:.3f}m) - Saving Model!")
                            best_cte = cte
                            best_completion = completion
                            torch.save(model.state_dict(), "model_checkpoints_sparse/carla_vla_furthest_driver_gated.pth")
                        else:
                            print(f"CTE NOT IMPROVED ({cte:.3f}m vs {best_cte:.3f}m) - Not Saving")
                    else:
                        best_completion = completion
                        torch.save(model.state_dict(), "model_checkpoints_sparse/carla_vla_furthest_driver_gated.pth")
                        print(f">>> NEW FURTHEST DRIVER ({completion:.1f}%)")

                if completion > 98.0 and cte < 1.0 and easy_eval:
                    print(f">>> PERFECT DRIVER ON EVAL ROUTES FOUND! Changing eval to harder routes...")
                    ROUTES_TO_EVAL = [(31, 99), (13, 34), (72, 106)]
                    ROUTES_EVAL_NAMES = {
                        (31, 99): "Hard 1",
                        (13, 34): "Hard 2",
                        (72, 106): "Hard 3",
                    }
                    easy_eval = False

            print(print_str)

            if use_wandb:
                if not args.no_eval:
                    wandb_dict = {"Total Completion": completion, "Avg CTE": cte}
                    wandb_dict.update(route_results)
                    wandb.log(wandb_dict, step=epoch)
                torch.save(model.state_dict(), f"model_checkpoints_sparse/carla_vla_epoch_{epoch}_gated.pth")
                wandb.save(f"model_checkpoints_sparse/carla_vla_epoch_{epoch}_gated.pth")

if __name__ == "__main__":
    train()
