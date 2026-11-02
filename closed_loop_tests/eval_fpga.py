import carla
import sys
import os
import queue
import pandas as pd
import subprocess
import time
import signal
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
import random
import gc
import cv2
import math
from models.carla_model import CarlaVLAHW as CarlaVLA
from fpga_client.fpga_agent_client import FPGAAgentClient
from carla_utils.carla_server import CarlaServer
from dataloaders.carla_dataloaders import CarlaImageCache, CarlaFastDataset, collate_fn
from dataloaders.carla_augmentations import GPUAugmenter
from carla_utils.route_utils import get_closest_waypoint_index, calculate_route_length, get_enhanced_command, command_to_branch_id

sys.path.append("/home/eeafzal/carla_simulator/PythonAPI/carla")
from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.local_planner import RoadOption

BASE_DATASET_DIR = "carla_dataset_base"
CSV_PATH = os.path.join(BASE_DATASET_DIR, "data.csv")
IMG_DIR = os.path.join(BASE_DATASET_DIR, "images")

USE_FPGA = True
FPGA_SERVER_IP_ADDR = "143.89.78.219"
FPGA_SERVER_PORT = 12345

CARLA_CLIENT_PORT = 3000
EVAL_SCENARIO = "base"
N_VEHICLES = 40
N_WALKERS = 40

BATCH_SIZE = 1 
NUM_WORKERS = 1
PREFETCH_FACTOR = 1

STACK_FRAMES = 1
EVAL_RENDER_SIZE = (512, 512)
FOV = 110
IMG_SIZE = 256

USE_AUX_CAMERA_FOR_EVAL = False
AUX_CAMERA_SIZE = (1280, 720)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_in_carla(model, device, agent, save_to_wandb=False):
    model.eval()
    video_name = f"eval_fpga.mp4"

    carla_proc = start_carla_server()
    if carla_proc is None:
        print("Failed to start CARLA server. Skipping eval...")
        return 0.0, 100.0
    
    # Metrics State
    total_cte = 0.0
    frame_count = 0
    collision_occurred = False
    off_road_occurred = False
    
    try:
        client = carla.Client('localhost', CARLA_CLIENT_PORT)
        client.set_timeout(20.0)
        world = client.load_world('Town10HD_Opt')
        
        # Cleanup
        for actor in world.get_actors().filter('vehicle.*'): actor.destroy()
        for actor in world.get_actors().filter('sensor.*'): actor.destroy()
        
        # Physics Settings (Match Training)
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        settings.substepping = True
        settings.max_substep_delta_time = 0.01
        settings.max_substeps = 10
        world.apply_settings(settings)

        if EVAL_SCENARIO == "dynamic":
            pass
        elif EVAL_SCENARIO == "weather":
            weather = carla.WeatherParameters.HardRainNoon
            weather.fog_density = 10.0
            weather.wetness = 100.0
            world.set_weather(weather)
        elif EVAL_SCENARIO == "base":
            weather = carla.WeatherParameters.ClearNoon
            world.set_weather(weather)

        map_data = world.get_map()
        grp = GlobalRoutePlanner(map_data, sampling_resolution=2.0)
        
        spawn_points = map_data.get_spawn_points()
        start_pose = spawn_points[50]
        dest_pose = spawn_points[76]
        # start_pose = spawn_points[131]
        # dest_pose = spawn_points[145]
        
        # Calculate Total Route Length for Completion %
        full_route = grp.trace_route(start_pose.location, dest_pose.location)
        total_route_length = calculate_route_length(full_route)
        
        # Spawn
        bp = world.get_blueprint_library().find('vehicle.tesla.model3')
        vehicle = world.spawn_actor(bp, start_pose)
        
        # Sensors
        cam_bp = world.get_blueprint_library().find('sensor.camera.rgb')
        cam_bp.set_attribute('image_size_x', str(EVAL_RENDER_SIZE[0]))
        cam_bp.set_attribute('image_size_y', str(EVAL_RENDER_SIZE[1]))
        cam_bp.set_attribute('fov', str(FOV))
        camera = world.spawn_actor(cam_bp, carla.Transform(carla.Location(x=1.5, z=2.4)), attach_to=vehicle)

        if USE_AUX_CAMERA_FOR_EVAL:
            aux_bp = world.get_blueprint_library().find('sensor.camera.rgb')
            aux_bp.set_attribute('image_size_x', str(AUX_CAMERA_SIZE[0]))
            aux_bp.set_attribute('image_size_y', str(AUX_CAMERA_SIZE[1]))
            aux_bp.set_attribute('fov', str(FOV))
            aux_camera = world.spawn_actor(aux_bp, carla.Transform(carla.Location(x=1.5, z=2.4)), attach_to=vehicle)
            # We won't use the aux camera for inference, but we can record it for diagnostics if needed.
            aux_q = queue.Queue()
            aux_camera.listen(aux_q.put)
            aux_video_name = f"aux_eval_fpga.mp4"
            aux_video_out = cv2.VideoWriter(aux_video_name, cv2.VideoWriter_fourcc(*'mp4v'), 20, (AUX_CAMERA_SIZE[0], AUX_CAMERA_SIZE[1]))

        col_bp = world.get_blueprint_library().find('sensor.other.collision')
        collision_sensor = world.spawn_actor(col_bp, carla.Transform(), attach_to=vehicle)
        
        q = queue.Queue()
        camera.listen(q.put)
        
        col_q = queue.Queue()
        collision_sensor.listen(col_q.put)
        
        video_out = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), 20, (EVAL_RENDER_SIZE[0], EVAL_RENDER_SIZE[1]))

        def clear_q(q):
            while not q.empty():
                try: q.get_nowait()
                except queue.Empty: break

        clear_q(q)
        if USE_AUX_CAMERA_FOR_EVAL: clear_q(aux_q)

        # Inference State
        frame_stack = deque(maxlen=STACK_FRAMES)
        prev_action = torch.zeros(1, 3).to(device) # [Steer, Accel, Speed]
        current_route = list(full_route) # Copy
        
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

        for _ in range(30): 
            world.tick()
            try: q.get(block=False, timeout=1.0)
            except queue.Empty: pass
            if USE_AUX_CAMERA_FOR_EVAL:
                try: aux_q.get(block=False, timeout=1.0)
                except queue.Empty: pass
        world.tick()

        vehicle.apply_control(carla.VehicleControl(manual_gear_shift=True, gear=1))
        for _ in range(5): world.tick()
        vehicle.apply_control(carla.VehicleControl(throttle=1.0))
        for _ in range(10): world.tick()
        vehicle.apply_control(carla.VehicleControl(throttle=0.4, manual_gear_shift=False, hand_brake=False))
        clear_q(q)
        if USE_AUX_CAMERA_FOR_EVAL: clear_q(aux_q)
        for _ in range(50):
            world.tick()
            try: q.get(block=False, timeout=1.0)
            except queue.Empty: pass
            if USE_AUX_CAMERA_FOR_EVAL:
                try: aux_q.get(block=False, timeout=1.0)
                except queue.Empty: pass
        world.tick()

        print(f"Evaluating...")
        stop_counter = 0

        full_route = grp.trace_route(start_pose.location, dest_pose.location)
        total_route_length = 0.0
        cumulative_distances = [0.0]

        for i in range(len(full_route) - 1):
            # Calculate 2D distance between waypoints
            loc1 = full_route[i][0].transform.location
            loc2 = full_route[i+1][0].transform.location
            # Use 2D distance (X, Y) as requested
            d = math.sqrt((loc1.x - loc2.x)**2 + (loc1.y - loc2.y)**2)
            total_route_length += d
            cumulative_distances.append(total_route_length)

        # Progress State
        max_reached_idx = 0
        
        for f in range(5000):
            ego_loc = vehicle.get_location()
            ego_yaw = vehicle.get_transform().rotation.yaw
            
            # Check Collision
            if not col_q.empty():
                print("Collision detected!")
                collision_occurred = True
                if USE_FPGA:
                    agent.get_action(None, 0, None, None, 1) # Terminate server
                break
            
            carla_img = q.get(block=True, timeout=2.0)
            if USE_AUX_CAMERA_FOR_EVAL:
                aux_img = aux_q.get(block=True, timeout=2.0)
            
            closest_idx, dist_to_route = get_closest_waypoint_index(ego_loc, full_route)
            
            total_cte += dist_to_route
            frame_count += 1

            ego_wp = map_data.get_waypoint(ego_loc, project_to_road=False, lane_type=carla.LaneType.Any)
            if ego_wp:
                if ego_wp.lane_type in [carla.LaneType.Sidewalk]:
                    print(f"Failed: Went Off-Road ({ego_wp.lane_type})")
                    off_road_occurred = True
                    if USE_FPGA:
                        agent.get_action(None, 0, None, None, 1) # Terminate server
                    break
            
            # Failure Condition: Drifting too far
            if dist_to_route > 20.0:
                print(f"Failed: CTE too high ({dist_to_route:.2f}m)")
                if USE_FPGA:
                    agent.get_action(None, 0, None, None, 1) # Terminate server
                break
                
            if len(current_route) == 0 or ego_loc.distance(dest_pose.location) < 5.0:
                print("Success: Destination Reached!")
                if USE_FPGA:
                    agent.get_action(None, 0, None, None, 1) # Terminate server
                completion_ratio = 100.0
                break
            
            #Check if stuck (Low velocity and no progress for 200 consecutive frames and not stopped at a red light)
            v_vec = vehicle.get_velocity()
            speed_ms = math.sqrt(v_vec.x**2 + v_vec.y**2 + v_vec.z**2)
            if speed_ms < 0.5 and check_hazards(vehicle, world, map_data) is None:
                stop_counter += 1
                if stop_counter > 100:
                    print("Failed: Stuck for too long")
                    if USE_FPGA:
                        agent.get_action(None, 0, None, None, 1) # Terminate server
                    break
            else:
                stop_counter = 0

            array = np.frombuffer(carla_img.raw_data, dtype=np.uint8)
            array = np.reshape(array, (EVAL_RENDER_SIZE[1], EVAL_RENDER_SIZE[0], 4))[:, :, :3]
            video_frame = array.copy()
            if USE_AUX_CAMERA_FOR_EVAL:
                aux_array = np.frombuffer(aux_img.raw_data, dtype=np.uint8)
                aux_array = np.reshape(aux_array, (AUX_CAMERA_SIZE[1], AUX_CAMERA_SIZE[0], 4))[:, :, :3]
                aux_frame = aux_array.copy()
            ah, aw = array.shape[:2]
            side = min(ah, aw)
            start_x = (aw - side) // 2
            start_y = (ah - side) // 2
            array = array[start_y:start_y+side, start_x:start_x+side]
            array = array.copy()
            res = cv2.resize(array, (256, 256), interpolation=cv2.INTER_AREA)
            resized = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(resized).permute(2, 0, 1).float().to(device) / 255.0
            
            if len(frame_stack)==0: 
                for _ in range(STACK_FRAMES): frame_stack.append(tensor)
            else: frame_stack.append(tensor)
            
            input_stack = torch.cat(list(frame_stack), dim=0).unsqueeze(0)
            input_stack = (input_stack.reshape(-1, 3, 256, 256) - mean) / std
            input_stack = input_stack.reshape(1, 3 * STACK_FRAMES, 256, 256)
            
            search_len = min(len(current_route), 20)
            closest_idx = 0
            min_dist = float('inf')

            for i in range(search_len):
                d = current_route[i][0].transform.location.distance(ego_loc)
                if d < min_dist:
                    min_dist = d
                    closest_idx = i

            if min_dist < 15.0: 
                current_route = current_route[closest_idx:]
            
            # Command
            hazard_msg = check_hazards(vehicle, world, map_data)

            if hazard_msg: instruction = hazard_msg
            else: instruction = get_enhanced_command(current_route, ego_loc, ego_yaw)

            _, dist_to_center = get_closest_waypoint_index(ego_loc, current_route)
            is_mid_maneuver = dist_to_center > 0.5
            if not is_mid_maneuver:
                if instruction == "change lane left":
                    if not is_lane_safe(vehicle, world, map_data, 'left'):
                        instruction = "follow lane"
                elif instruction == "change lane right":
                    if not is_lane_safe(vehicle, world, map_data, 'right'):
                        instruction = "follow lane"

            branch_id = command_to_branch_id(instruction)

            if not USE_FPGA:
                with torch.no_grad():
                    pred = model(input_stack, prev_action)
                    pred = pred[0, branch_id]
                s = pred[0].item()
                accel = pred[1].item()
            if USE_FPGA:
                input_stack = input_stack.detach().cpu()[0]
                in_max = torch.max(torch.abs(input_stack)).item()
                input_stack = input_stack.numpy()
                in_scale = 127 / in_max
                input_stack = (input_stack * in_scale).round().clip(-128, 127).astype(np.int8).flatten() 
                with torch.no_grad():
                    prev_action_features = F.relu(model.measure_proj(prev_action))
                    prev_action_features = (prev_action_features.detach().cpu().numpy()*1.804250271749319).round().clip(-128, 127).astype(np.int8).flatten()
                    s_fpga, a_fpga = agent.get_action(input_stack,
                                                    branch_id,
                                                    prev_action_features,
                                                    prev_action[0, :].detach().cpu().numpy(),
                                                    0)
                    s = s_fpga
                    accel = a_fpga
            
            # print(f"Steer GPU: {s:.3f} | Accel GPU: {accel:.3f} | Steer FPGA: {s_fpga:.3f} | Accel FPGA: {a_fpga:.3f} | CMD: {instruction}")
            # agent.get_action(None, 0, None, None, 1) # Terminate server
            # break

            # Convert Accel -> Throttle/Brake
            if accel > 0: t = accel; b = 0.0
            else: t = 0.0; b = -accel
            
            # Deadzones
            # if t < 0.1: t = 0.0
            if b < 0.1: b = 0.0
            
            # Speed Feedback
            v_vec = vehicle.get_velocity()
            speed_ms = math.sqrt(v_vec.x**2 + v_vec.y**2 + v_vec.z**2)
            prev_action = torch.tensor([s, accel, speed_ms]).unsqueeze(0).to(device)

            # Anti-Stall (Boost low throttle)
            # if speed_ms < 0.5 and t > 0.1: t = max(t, 0.6); b = 0.0
            if speed_ms < 0.1 and b < 0.1 and "stop" not in instruction:
                t = max(t, 0.3) # Force a kickstart

            vehicle.apply_control(carla.VehicleControl(steer=float(s), throttle=float(t), brake=float(b), manual_gear_shift=False))
            world.tick()

            ego_loc = vehicle.get_location()
        
            # Only look 20 waypoints ahead of where we last were
            search_start = max_reached_idx
            search_end = min(max_reached_idx + 20, len(full_route))
            
            best_dist = float('inf')
            best_idx = max_reached_idx

            for i in range(search_start, search_end):
                wp_loc = full_route[i][0].transform.location
                # 2D distance to waypoint
                d = math.sqrt((ego_loc.x - wp_loc.x)**2 + (ego_loc.y - wp_loc.y)**2)
                
                # We only count a waypoint as "reached" if we are reasonably close to it
                # (e.g., within 5 meters) to prevent jumping through buildings
                if d < best_dist and d < 5.0:
                    best_dist = d
                    best_idx = i

            # Update the furthest progress we've ever made
            if best_idx > max_reached_idx:
                max_reached_idx = best_idx

            # Calculate Completion based on pre-calculated route length
            current_progress_m = cumulative_distances[max_reached_idx]
            completion_ratio = (current_progress_m / total_route_length) * 100.0
            
            # UI
            # cv2.putText(resized, f"CTE: {dist_to_route:.2f}m", (10, 20), 0, 0.5, (0, 255, 255), 1)
            video_frame = draw_full_diagnostic(video_frame, [s, t, b], cmd=instruction)
            video_out.write(video_frame)
            if USE_AUX_CAMERA_FOR_EVAL:
                aux_frame = draw_full_diagnostic(aux_frame, [s, t, b], cmd=instruction)
                aux_video_out.write(aux_frame)
    
        avg_cte = total_cte / frame_count if frame_count > 0 else 0.0
        
        print(f"EVAL RESULT: Completion={completion_ratio:.1f}% | Avg CTE={avg_cte:.3f}m | Collision={collision_occurred} | Off-road={off_road_occurred}")
        print(f"Eval video saved to {video_name}")
        
        return completion_ratio, avg_cte

    except Exception as e:
        print(f"Eval Error: {e}")
        return 0.0, 100.0
    finally:
        video_out.release()
        if USE_AUX_CAMERA_FOR_EVAL: aux_video_out.release()
        if 'aux_camera' in locals(): aux_camera.destroy()
        if 'camera' in locals(): camera.destroy()
        if 'collision_sensor' in locals(): collision_sensor.destroy()
        if 'vehicle' in locals(): vehicle.destroy()
        stop_carla_server(carla_proc)
        model.train()
        if save_to_wandb:
            import wandb
            wandb.log({f"Eval Standalone": wandb.Video(video_name, format="mp4")})


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model checkpoint for evaluation')
    parser.add_argument('--sample_idx', type=int, default=224, help='Use fast in-RAM dataset (requires more RAM)')
    args = parser.parse_args()

    model_path = args.model_path
    sample_idx = args.sample_idx

    print(f"Reading CSV from {CSV_PATH}...")
    full_df = pd.read_csv(CSV_PATH)
    full_df = full_df.iloc[sample_idx:sample_idx+1, :].reset_index(drop=True)
    full_df['image_path'] = os.path.join(BASE_DATASET_DIR, "images") + "/" + full_df['image_path']
    print(full_df)

    model = CarlaVLA(STACK_FRAMES).to(DEVICE)

    sd = torch.load(model_path, map_location='cpu')
    model.eval()
    model.load_state_dict(sd, strict=True)

    valid_df = full_df
    # valid_df = valid_df.iloc[:500, :].reset_index(drop=True)
    
    train_indices = valid_df.index.tolist()
    
    print(f"Eval Frames: {len(train_indices)}")
    
    print("Using Fast In-RAM Dataset...")
    cache_manager = CarlaImageCache(valid_df, IMG_SIZE)
    eval_ds = CarlaFastDataset(cache_manager, train_indices, is_train=False)
    
    train_loader = DataLoader(eval_ds, batch_size=BATCH_SIZE, shuffle=False, 
                              num_workers=NUM_WORKERS, pin_memory=True, drop_last=False,
                              prefetch_factor=PREFETCH_FACTOR, collate_fn=collate_fn)
    
    augmenter = GPUAugmenter().to(DEVICE) 
    
    agent = None
    if USE_FPGA:
        agent = FPGAAgentClient(FPGA_SERVER_IP_ADDR, FPGA_SERVER_PORT)
    model.eval()
    # evaluate_in_carla(model, DEVICE, agent, save_to_wandb=False)

    for imgs, branch_ids, prevs, targets in train_loader:

        imgs = imgs.to(DEVICE, non_blocking=True)
        branch_ids = branch_ids.to(DEVICE, non_blocking=True)
        prevs = prevs.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True)

        print(f"Previous Action: {prevs}")
        # prevs = torch.zeros_like(prevs)
        imgs = augmenter(imgs, is_train=False)

        # Save imgs to a txt file for debugging
        # np_imgs = imgs.detach().cpu().numpy()
        # np.savetxt("debug_imgs.txt", np_imgs.flatten())

        with torch.no_grad():
            preds = model(imgs, prevs)
        # print(preds.shape)
        gpu_preds = preds[0, 0].detach().cpu().numpy()
        gpu_steer = gpu_preds[0]
        gpu_accel = gpu_preds[1]
        print(f"GPU Pred: Steer={gpu_steer:.3f}, Accel={gpu_accel:.3f}")
        
        prevs = F.relu(model.measure_proj(prevs))
        prevs = (prevs.detach().cpu().numpy()*1.778922076038997).round().clip(-128, 127).astype(np.int8).flatten()
        
        # print(imgs[0, 0, :15, :15])
        imgs = imgs.detach().cpu()
        imgs = imgs[0]
        in_max = torch.max(torch.abs(imgs)).item()
        imgs = imgs.numpy()
        in_scale = 127 / in_max
        img_to_send = (imgs*in_scale).round().clip(-128, 127).astype(np.int8).flatten()
        steer, accel = agent.get_action(img_to_send, 0, prevs, (0, 0, 0), 0)
        print(f"FPGA Pred: Steer={steer:.3f}, Accel={accel:.3f}")
        # preds = model(imgs, prevs)
        # print(preds)
    agent.get_action(None, 0, None, None, 1) # Terminate server
 
if __name__ == "__main__":
    train()
