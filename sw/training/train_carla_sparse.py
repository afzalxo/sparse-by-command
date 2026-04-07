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
from carla_utils.eval_utils import evaluate_in_carla_gated, evaluate_in_carla_mp_wrapper, evaluate_single_route_wrapper
from carla_utils.carla_server import CarlaServer
from loss.branched_loss import branched_loss

sys.path.append("/path/to/carla/PythonAPI/carla")
from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.local_planner import RoadOption

BASE_DATASET_DIR = "/path/to/datasets/carla_dataset_base"
WEATHER_DATASET_DIR = "/path/to/datasets/carla_dataset_weather"
DYNAMIC_DATASET_DIR = "/path/to/datasets/carla_dataset_dynamic"

EVAL_SCENARIOS = ["dynamic"]
N_VEHICLES = 40
N_WALKERS = 40

LAMBDA_SPARSE = 0.002
TASK_DIM = 6

BATCH_SIZE = 256 
NUM_WORKERS = 8
PREFETCH_FACTOR = 2
N_TRAIN_SAMPLES = 0 # 420000
FLIP_PROB = 0.5

LEARNING_RATE = 2.0e-4
EPOCHS = 200
FINETUNE_EPOCHS = 50  # Used when --finetune is set (phases 2 and 3)
BEGIN_EVAL_EPOCH = 0

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
torch.cuda.manual_seed_all(42) 
torch.backends.cudnn.deterministic = True 
torch.backends.cudnn.benchmark = False


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval-only', action='store_true', help='Run evaluation only')
    parser.add_argument('--finetune', action='store_true', help='Finetune from a pretrained checkpoint')
    parser.add_argument('--model-path', type=str, required=False, help='Path to the model checkpoint for evaluation')
    parser.add_argument('--fast-dataset', action='store_true', help='Use fast in-RAM dataset (requires more RAM)')
    parser.add_argument('--use-wandb', action='store_true', help='Use Weights & Biases for logging')
    parser.add_argument('--no-eval', action='store_true', help='Disable evaluation during training')
    parser.add_argument('--resume', action='store_true', help='Resume training from the latest checkpoint')
    parser.add_argument('--phase', type=int, default=2, choices=[2, 3],
                        help='Training phase: 2=ramp lambda+soft masks (hard_masking=False), 3=peak lambda+hard masks (hard_masking=True)')
    parser.add_argument('--eval-gpus', type=int, nargs='+', default=[0],
                        help='GPU IDs for parallel evaluation (e.g. --eval-gpus 0 3 4). One process per route.')
    parser.add_argument("--use-fpga-emulator", action="store_true", help="Use INT8 FPGA emulator instead of GPU model")
    parser.add_argument("--n-vehicles", type=int, default=40, help="Number of NPC vehicles (0 for static eval)")
    args = parser.parse_args()

    phase = args.phase
    eval_gpus = args.eval_gpus
    hard_masking = (phase == 3)
    print(f"Training Phase {phase}: hard_masking={hard_masking}, eval_gpus={eval_gpus}")

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

    # df_base = process_df(BASE_DATASET_DIR, "base", 0, 0)
    # df_weather = process_df(WEATHER_DATASET_DIR, "weather", 1, 0)
    df_dynamic = process_df(DYNAMIC_DATASET_DIR, "dynamic", 0, 1)
    # print(f"Base Dataset: {len(df_base[df_base['is_train_candidate'] == 1])} frames, Weather Dataset: {len(df_weather[df_weather['is_train_candidate'] == 1])} frames, Dynamic Dataset: {len(df_dynamic[df_dynamic['is_train_candidate'] == 1])} frames")
    # full_df = pd.concat([df_base, df_weather, df_dynamic], ignore_index=True)
    full_df = df_dynamic

    full_df = full_df[full_df['is_train_candidate'] == 1].reset_index(drop=True)
    # if args.finetune:
    #     full_df = full_df[full_df['instruction'] == "follow lane"].reset_index(drop=True)

    model = CarlaVLA(task_dim=TASK_DIM).to(DEVICE)
    if args.finetune:
        # Phase 2: loading dense backbone (no gater params) → strict=False
        # Phase 3: loading full gated model → strict=True
        model.load_state_dict(torch.load(args.model_path), strict=(phase == 3))
        print(f"Loaded model from {args.model_path} for finetuning (strict={phase == 3}).")

    ROUTES_TO_EVAL = [(81, 139), (86, 4), (66, 62)]  # Easy routes (default)
    ROUTES_EVAL_NAMES = {
        (81, 139): "Left Turn",
        (86, 4): "Right Turn",
        (66, 62): "Lane Following",
    }
    start_epoch = 0
    easy_eval = True

    if args.resume or eval_only:
        model.load_state_dict(torch.load(model_path), strict=False)
        print(f"Loaded model from {model_path} for {'evaluation' if eval_only else 'resuming training'}.")
        start_epoch = 0

    # Use hard routes whenever starting from an existing trained checkpoint
    if args.finetune or args.resume or eval_only:
        ROUTES_TO_EVAL = [(31, 99), (13, 34), (72, 106)]
        ROUTES_EVAL_NAMES = {
            (31, 99): "Hard 1",
            (13, 34): "Hard 2",
            (72, 106): "Hard 3",
        }
        easy_eval = False
        print("Using hard evaluation routes.")

    if eval_only:
        ctx = multiprocessing.get_context('spawn')
        if len(eval_gpus) > 1:
            queues = []
            processes = []
            for i, route_indices in enumerate(ROUTES_TO_EVAL):
                gpu_id = eval_gpus[i % len(eval_gpus)]
                route_device = torch.device(f"cuda:{gpu_id}")
                route_name = ROUTES_EVAL_NAMES.get(route_indices, f"{route_indices[0]}->{route_indices[1]}")
                q = ctx.Queue()
                queues.append(q)
                p = ctx.Process(target=evaluate_single_route_wrapper,
                                args=(model_path, 0, route_device, EVAL_SCENARIOS,
                                      route_indices, route_name, EVAL_RENDER_SIZE, FOV,
                                      True, False, gpu_id, hard_masking, q))
                processes.append(p)
            for p in processes:
                p.start()
                time.sleep(3)
            for p in processes:
                p.join()
            total_rc, total_rl, total_cte, total_fc = 0.0, 0.0, 0.0, 0
            route_results = {}
            for q in queues:
                try:
                    rc, rl, cte_sum, fc, rr = q.get(timeout=30)
                    total_rc += rc; total_rl += rl; total_cte += cte_sum; total_fc += fc
                    route_results.update(rr)
                except Exception as e:
                    print(f"Failed to get eval result: {e}")
                    total_rl = max(total_rl, 1.0); total_fc = max(total_fc, 1)
            completion = (total_rc / total_rl) * 100.0 if total_rl > 0 else 0.0
            cte = total_cte / total_fc if total_fc > 0 else 0.0
        else:
            result_queue = ctx.Queue()
            p = ctx.Process(target=evaluate_in_carla_mp_wrapper,
                            args=(model_path, 0, torch.device(f"cuda:{eval_gpus[0]}"), EVAL_SCENARIOS, ROUTES_TO_EVAL,
                                  ROUTES_EVAL_NAMES, EVAL_RENDER_SIZE, FOV, True, False,
                                  result_queue, eval_gpus[0], hard_masking, args.use_fpga_emulator, args.n_vehicles))
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

    effective_epochs = FINETUNE_EPOCHS if args.finetune else EPOCHS
    begin_eval_epoch = 10 if args.finetune else BEGIN_EVAL_EPOCH
    print(f"Training for {effective_epochs} epochs, eval starts at epoch {begin_eval_epoch}.")

    best_completion = 0.0
    best_cte = float('inf')
    for epoch in range(start_epoch, effective_epochs):
        model.train()
        total_loss = 0
        total_losses = [0, 0]
        total_sparsity_loss = 0
        current_dropout = ACTION_DROPOUT_PROB

        if phase == 2:
            # Ramp up lambda over the full training budget
            current_lambda = LAMBDA_SPARSE * min(1.0, epoch / effective_epochs)
        else:
            # Phase 3: always use peak lambda with hard masks
            current_lambda = LAMBDA_SPARSE

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{effective_epochs}")
        
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
                preds, masks = model(imgs, prevs, task_vecs, hard_masking=hard_masking)
                driving_loss, all_losses = criterion(preds, targets, branch_ids)
                sparsity_loss = masks.mean()
                loss = driving_loss + current_lambda * sparsity_loss
            
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

        if epoch >= begin_eval_epoch:
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
                if len(eval_gpus) > 1:
                    queues = []
                    processes = []
                    for i, route_indices in enumerate(ROUTES_TO_EVAL):
                        gpu_id = eval_gpus[i % len(eval_gpus)]
                        route_device = torch.device(f"cuda:{gpu_id}")
                        route_name = ROUTES_EVAL_NAMES.get(route_indices, f"{route_indices[0]}->{route_indices[1]}")
                        q = ctx.Queue()
                        queues.append(q)
                        proc = ctx.Process(target=evaluate_single_route_wrapper,
                                           args=(tmp_weights_path, epoch, route_device, EVAL_SCENARIOS,
                                                 route_indices, route_name, EVAL_RENDER_SIZE, FOV,
                                                 True, False, gpu_id, hard_masking, q))
                        processes.append(proc)
                    for proc in processes:
                        proc.start()
                        time.sleep(3)
                    for proc in processes:
                        proc.join()
                    total_rc, total_rl, total_cte, total_fc = 0.0, 0.0, 0.0, 0
                    route_results = {}
                    for q in queues:
                        try:
                            rc, rl, cte_sum, fc, rr = q.get(timeout=30)
                            total_rc += rc; total_rl += rl; total_cte += cte_sum; total_fc += fc
                            route_results.update(rr)
                        except Exception as e:
                            print(f"Failed to get eval result: {e}")
                            total_rl = max(total_rl, 1.0); total_fc = max(total_fc, 1)
                    completion = (total_rc / total_rl) * 100.0 if total_rl > 0 else 0.0
                    cte = total_cte / total_fc if total_fc > 0 else 0.0
                else:
                    result_queue = ctx.Queue()
                    p = ctx.Process(target=evaluate_in_carla_mp_wrapper,
                                    args=(tmp_weights_path, epoch, DEVICE, EVAL_SCENARIOS, ROUTES_TO_EVAL,
                                          ROUTES_EVAL_NAMES, EVAL_RENDER_SIZE, FOV, True, False,
                                          result_queue, eval_gpus[0], hard_masking))
                    p.start()
                    time.sleep(5)
                    p.join()
                    completion, cte, route_results = result_queue.get()

                if os.path.exists(tmp_weights_path):
                    os.remove(tmp_weights_path)

                print_str += f" | Completion: {completion:.2f}% | Avg CTE: {cte:.3f}m"

                best_ckpt_path = f"model_checkpoints_sparse/carla_vla_furthest_driver_phase{phase}_gated.pth"
                if completion >= best_completion and not easy_eval:
                    if completion == best_completion:
                        print(f">>> TIE IN COMPLETION ({completion:.1f}%) - ", end="")
                        if cte < best_cte:
                            print(f"NEW BEST CTE ({cte:.3f}m vs {best_cte:.3f}m) - Saving Model!")
                            best_cte = cte
                            best_completion = completion
                            torch.save(model.state_dict(), best_ckpt_path)
                            torch.save(model.state_dict(), f"model_checkpoints_sparse/carla_vla_phase{phase}_epoch{epoch}_comp{completion:.0f}_cte{cte:.3f}.pth")
                        else:
                            print(f"CTE NOT IMPROVED ({cte:.3f}m vs {best_cte:.3f}m) - Not Saving")
                    else:
                        if cte < best_cte:
                            best_cte = cte
                        best_completion = completion
                        torch.save(model.state_dict(), best_ckpt_path)
                        torch.save(model.state_dict(), f"model_checkpoints_sparse/carla_vla_phase{phase}_epoch{epoch}_comp{completion:.0f}_cte{cte:.3f}.pth")
                        print(f">>> NEW FURTHEST DRIVER ({completion:.1f}%), CTE ({cte:.3f})")

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
