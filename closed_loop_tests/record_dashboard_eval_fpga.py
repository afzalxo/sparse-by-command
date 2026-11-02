"""
Record a CARLA evaluation with FPGA inference for the dashboard video.

Usage (on GPU server, with FPGA server already running):
    python3 -u record_dashboard_eval_fpga.py \
        --model-path model_checkpoints_sparse/carla_vla_furthest_driver_phase3_gated.pth \
        --sim-gpu 3 --spawn 13 --dest 34 \
        --fpga-host 143.89.78.219 --fpga-port 12345 \
        --out-dir dashboard_data_fpga

The model is loaded on CPU only (for gater mask computation + quantization).
Inference goes to the FPGA via TCP.
"""
import argparse
import json
import math
import os
import queue
import random
import time
import socket
from collections import deque

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import carla

import sys
sys.path.append("/home/eeafzal/carla_simulator/PythonAPI/carla")
from agents.navigation.global_route_planner import GlobalRoutePlanner

from carla_utils.carla_server import CarlaServer
from carla_utils.route_utils import (
    get_closest_waypoint_index, calculate_route_length,
    check_hazards, is_lane_safe, get_enhanced_command, command_to_branch_id
)
from carla_utils.visualization_utils import draw_full_diagnostic
from carla_utils.traffic_utils import find_free_port, spawn_traffic
from models.carla_model import CarlaVLAHW_Gated
from fpga_client.fpga_agent_client import FPGAAgentClient


def record_eval(args):
    RENDER_SIZE = (1920, 1080)
    FOV = 110

    os.makedirs(args.out_dir, exist_ok=True)

    # ── Deterministic seeding (matches evaluate_single_route_wrapper) ─────
    route_seed = args.spawn * 1000 + args.dest
    random.seed(route_seed)
    np.random.seed(route_seed)

    # ── Start CARLA on sim GPU ────────────────────────────────────────────
    client_port = find_free_port(30000 + args.sim_gpu * 1000)
    carla_server = CarlaServer(gpu_rank=args.sim_gpu, rpc_port=client_port)
    carla_server.start()

    client = carla.Client('localhost', client_port)
    client.set_timeout(60.0)
    client.load_world('Town10HD_Opt')
    time.sleep(5)

    world = client.get_world()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    settings.substepping = True
    settings.max_substep_delta_time = 0.01
    settings.max_substeps = 10
    world.apply_settings(settings)
    world.set_weather(carla.WeatherParameters.ClearNoon)

    # Spawn dynamic traffic (matches training eval)
    for _ in range(10):
        world.tick()
    traffic_actors = spawn_traffic(world, client, n_vehicles=40, n_walkers=40,
                                   gpu_rank=args.sim_gpu)

    # ── Load model on CPU (for gater + quantization only) ─────────────────
    model = CarlaVLAHW_Gated(task_dim=6, oc_par=16)
    sd = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(sd, strict=True)
    model.eval()

    predictable_tiles = list(model.predictable_tiles)
    total_tiles_per_layer = list(model.total_tiles_per_layer)

    # ── Connect to FPGA server ────────────────────────────────────────────
    agent = FPGAAgentClient(args.fpga_host, args.fpga_port)

    # ── Spawn vehicle + sensors ───────────────────────────────────────────
    for actor in world.get_actors().filter('vehicle.*'):
        if actor.attributes.get('role_name') != 'autopilot':
            actor.destroy()
    for actor in world.get_actors().filter('sensor.*'):
        actor.destroy()

    map_data = world.get_map()
    grp = GlobalRoutePlanner(map_data, sampling_resolution=2.0)

    spawn_points = map_data.get_spawn_points()
    start_pose = spawn_points[args.spawn]
    dest_pose = spawn_points[args.dest]

    full_route = grp.trace_route(start_pose.location, dest_pose.location)

    cumulative_distances = [0.0]
    total_route_length = 0.0
    for i in range(len(full_route) - 1):
        loc1 = full_route[i][0].transform.location
        loc2 = full_route[i + 1][0].transform.location
        d = math.sqrt((loc1.x - loc2.x) ** 2 + (loc1.y - loc2.y) ** 2)
        total_route_length += d
        cumulative_distances.append(total_route_length)

    bp = world.get_blueprint_library().find('vehicle.tesla.model3')
    vehicle = world.spawn_actor(bp, start_pose)

    cam_bp = world.get_blueprint_library().find('sensor.camera.rgb')
    cam_bp.set_attribute('image_size_x', str(RENDER_SIZE[0]))
    cam_bp.set_attribute('image_size_y', str(RENDER_SIZE[1]))
    cam_bp.set_attribute('fov', str(FOV))
    camera = world.spawn_actor(
        cam_bp, carla.Transform(carla.Location(x=1.5, z=2.4)), attach_to=vehicle)

    col_bp = world.get_blueprint_library().find('sensor.other.collision')
    collision_sensor = world.spawn_actor(col_bp, carla.Transform(), attach_to=vehicle)

    img_q = queue.Queue()
    camera.listen(img_q.put)
    col_q = queue.Queue()
    collision_sensor.listen(col_q.put)

    def clear_q(q):
        while not q.empty():
            try:
                q.get_nowait()
            except queue.Empty:
                break

    # Let sim settle
    clear_q(img_q)
    for _ in range(30):
        world.tick()
        try:
            img_q.get(block=False, timeout=1.0)
        except queue.Empty:
            pass
    world.tick()

    # Kickstart vehicle
    vehicle.apply_control(carla.VehicleControl(manual_gear_shift=True, gear=1))
    for _ in range(5):
        world.tick()
    vehicle.apply_control(carla.VehicleControl(throttle=1.0))
    for _ in range(10):
        world.tick()
    vehicle.apply_control(carla.VehicleControl(throttle=0.4, manual_gear_shift=False, hand_brake=False))
    clear_q(img_q)
    for _ in range(50):
        world.tick()
        try:
            img_q.get(block=False, timeout=1.0)
        except queue.Empty:
            pass
    world.tick()

    # ── Video writer ──────────────────────────────────────────────────────
    vid_path = os.path.join(args.out_dir, "eval_video.mp4")
    vid_writer = cv2.VideoWriter(vid_path, cv2.VideoWriter_fourcc(*'mp4v'),
                                 20, RENDER_SIZE)

    # ── Inference loop ────────────────────────────────────────────────────
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    prev_action = torch.zeros(1, 3)
    current_route = list(full_route)
    frame_stack = deque(maxlen=1)
    telemetry = []

    max_reached_idx = 0
    stop_counter = 0
    total_energy_j = 0.0

    # Get the cumulative scale for measure_proj quantization
    # (this is hardcoded in eval_fpga.py as 1.804250271749319, but let's compute it)
    # We'll use the same hardcoded value for consistency with the FPGA host
    MEASURE_PROJ_SCALE = 43.163602387878953

    print(f"Recording FPGA eval  route {args.spawn}->{args.dest} ...")

    try:
      for f_idx in range(5000):
        ego_loc = vehicle.get_location()
        ego_yaw = vehicle.get_transform().rotation.yaw

        if not col_q.empty():
            print("Collision — ending.")
            break

        carla_img = img_q.get(block=True, timeout=2.0)

        closest_idx, dist_to_route = get_closest_waypoint_index(ego_loc, full_route)
        ego_wp = map_data.get_waypoint(ego_loc, project_to_road=False,
                                       lane_type=carla.LaneType.Any)
        if ego_wp and ego_wp.lane_type in [carla.LaneType.Sidewalk]:
            print("Off-road — ending.")
            break
        if dist_to_route > 20.0:
            print("CTE too high — ending.")
            break
        if ego_loc.distance(dest_pose.location) < 5.0:
            print("Destination reached!")
            break

        v_vec = vehicle.get_velocity()
        speed_ms = math.sqrt(v_vec.x ** 2 + v_vec.y ** 2 + v_vec.z ** 2)
        if speed_ms < 0.5 and check_hazards(vehicle, world, map_data) is None:
            stop_counter += 1
            if stop_counter > 100:
                print("Stuck — ending.")
                break
        else:
            stop_counter = 0

        # ── Preprocess image ──────────────────────────────────────────────
        array = np.frombuffer(carla_img.raw_data, dtype=np.uint8)
        array = np.reshape(array, (RENDER_SIZE[1], RENDER_SIZE[0], 4))[:, :, :3]
        video_frame = array.copy()

        res = cv2.resize(array, (256, 256), interpolation=cv2.INTER_AREA)
        resized = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0

        if len(frame_stack) == 0:
            frame_stack.append(tensor)
        else:
            frame_stack.append(tensor)

        input_stack = torch.cat(list(frame_stack), dim=0).unsqueeze(0)
        input_stack = (input_stack.reshape(-1, 3, 256, 256) - mean) / std
        input_stack = input_stack.reshape(1, 3, 256, 256)

        # ── Navigation command ────────────────────────────────────────────
        search_len = min(len(current_route), 20)
        ci = 0
        min_dist = float('inf')
        for i in range(search_len):
            d = current_route[i][0].transform.location.distance(ego_loc)
            if d < min_dist:
                min_dist = d
                ci = i
        if min_dist < 15.0:
            current_route = current_route[ci:]

        hazard_msg = check_hazards(vehicle, world, map_data)
        instruction = hazard_msg if hazard_msg else get_enhanced_command(current_route, ego_loc, ego_yaw)

        _, dist_to_center = get_closest_waypoint_index(ego_loc, current_route)
        is_mid_maneuver = dist_to_center > 0.5
        if not is_mid_maneuver:
            if instruction == "change lane left" and not is_lane_safe(vehicle, world, map_data, 'left'):
                instruction = "follow lane"
            elif instruction == "change lane right" and not is_lane_safe(vehicle, world, map_data, 'right'):
                instruction = "follow lane"

        branch_id = command_to_branch_id(instruction)
        task_vec = torch.zeros(1, 6, dtype=torch.float32)
        task_vec[0, branch_id] = 1.0

        # ── Quantize inputs for FPGA ─────────────────────────────────────
        img_for_fpga = input_stack.detach().cpu()[0]
        in_max = torch.max(torch.abs(img_for_fpga)).item()
        in_scale = 127.0 / in_max
        img_int8 = (img_for_fpga.numpy() * in_scale).round().clip(-128, 127).astype(np.int8).flatten()

        with torch.no_grad():
            prev_feat = F.relu(model.measure_proj(prev_action))
            prev_feat_int8 = (prev_feat.numpy() * MEASURE_PROJ_SCALE).round().clip(-128, 127).astype(np.int8).flatten()

        # ── FPGA inference ────────────────────────────────────────────────
        s_fpga, a_fpga, fpga_lat_ms, fpga_power_w = agent.get_action(
            img_int8, branch_id, prev_feat_int8, None, 0)

        s = s_fpga
        accel = a_fpga
        latency_ms = fpga_lat_ms if fpga_lat_ms else 0.0
        fpga_power = fpga_power_w if fpga_power_w else 0.0

        # Energy accumulation
        total_energy_j += fpga_power * (latency_ms / 1000.0)

        t_ctrl = max(accel, 0.0)
        b_ctrl = max(-accel, 0.0)
        if b_ctrl < 0.1:
            b_ctrl = 0.0
        if speed_ms < 0.1 and b_ctrl < 0.1 and "stop" not in instruction:
            t_ctrl = max(t_ctrl, 0.3)

        prev_action = torch.tensor([s, accel, speed_ms]).unsqueeze(0)
        vehicle.apply_control(carla.VehicleControl(
            steer=float(s), throttle=float(t_ctrl), brake=float(b_ctrl), manual_gear_shift=False))
        world.tick()

        # ── Progress tracking ─────────────────────────────────────────────
        ego_loc = vehicle.get_location()
        search_start = max_reached_idx
        search_end = min(max_reached_idx + 20, len(full_route))
        best_dist = float('inf')
        best_idx = max_reached_idx
        for i in range(search_start, search_end):
            wp_loc = full_route[i][0].transform.location
            d = math.sqrt((ego_loc.x - wp_loc.x) ** 2 + (ego_loc.y - wp_loc.y) ** 2)
            if d < best_dist and d < 5.0:
                best_dist = d
                best_idx = i
        if best_idx > max_reached_idx:
            max_reached_idx = best_idx

        current_progress_m = cumulative_distances[max_reached_idx]
        completion_pct = (current_progress_m / total_route_length) * 100.0

        # ── Draw HUD + write video ────────────────────────────────────────
        video_frame = draw_full_diagnostic(video_frame, [s, t_ctrl, b_ctrl], cmd=instruction)
        vid_writer.write(video_frame)

        # ── Compute tile masks on CPU (for dashboard visualization) ───────
        with torch.no_grad():
            all_pred_masks = model.gater(task_vec)
        masks_np = all_pred_masks[0].numpy()

        total_active = 0
        total_all = 0
        offset = 0
        for li in range(len(total_tiles_per_layer)):
            n_tiles = total_tiles_per_layer[li]
            total_all += n_tiles
            total_active += 1  # tile 0
            pt = predictable_tiles[li]
            chunk = masks_np[offset:offset + pt]
            total_active += int((chunk > 0.5).sum())
            offset += pt
        sparsity_pct = (1.0 - total_active / total_all) * 100.0

        inferences_per_kj = (f_idx + 1) / (total_energy_j / 1000.0) if total_energy_j > 0 else 0.0

        # ── Record telemetry ──────────────────────────────────────────────
        telemetry.append({
            'frame_idx': f_idx,
            'latency_ms': round(latency_ms, 4),
            'gpu_power_w': round(fpga_power, 2),  # re-use field name for dashboard compat
            'idle_power_w': 0.0,
            'masks': masks_np.tolist(),
            'branch_id': branch_id,
            'instruction': instruction,
            'steer': round(s, 4),
            'accel': round(accel, 4),
            'speed_ms': round(speed_ms, 3),
            'completion_pct': round(completion_pct, 2),
            'sparsity_pct': round(sparsity_pct, 1),
            'inferences_per_kj': round(inferences_per_kj, 1),
            'total_energy_j': round(total_energy_j, 6),
        })

        if f_idx % 100 == 0:
            print(f"  Frame {f_idx:4d} | lat={latency_ms:.3f}ms | pwr={fpga_power:.1f}W | "
                  f"sprs={sparsity_pct:.1f}% | comp={completion_pct:.1f}% | cmd={instruction}")

    except Exception as e:
        print(f"Error during recording: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Send terminate to FPGA server
        try:
            agent.get_action(None, 0, None, None, 1)
        except Exception:
            pass

        vid_writer.release()
        if 'camera' in locals():
            camera.destroy()
        if 'collision_sensor' in locals():
            collision_sensor.destroy()
        if 'vehicle' in locals():
            vehicle.destroy()
        if 'traffic_actors' in locals():
            client.apply_batch([carla.command.DestroyActor(x) for x in traffic_actors])
        carla_server.stop()

    # Save telemetry
    telem_path = os.path.join(args.out_dir, "telemetry.json")
    with open(telem_path, 'w') as fp:
        json.dump({
            'predictable_tiles': predictable_tiles,
            'total_tiles_per_layer': total_tiles_per_layer,
            'idle_power_w': 0.0,
            'route': f"{args.spawn}->{args.dest}",
            'model_path': args.model_path,
            'sim_gpu': args.sim_gpu,
            'accelerator': 'fpga',
            'fpga_host': args.fpga_host,
            'frames': telemetry,
        }, fp)

    n = len(telemetry)
    avg_lat = np.mean([t['latency_ms'] for t in telemetry]) if n else 0
    avg_pwr = np.mean([t['gpu_power_w'] for t in telemetry]) if n else 0
    final_ipkj = telemetry[-1]['inferences_per_kj'] if n else 0

    print(f"\nDone. {n} frames recorded.")
    print(f"  Avg FPGA latency: {avg_lat:.3f} ms")
    print(f"  Avg FPGA power:   {avg_pwr:.1f} W")
    print(f"  Inf/kJ:           {final_ipkj:.0f}")
    print(f"  Video: {vid_path}")
    print(f"  Telemetry: {telem_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Record FPGA dashboard eval data")
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--sim-gpu', type=int, default=3, help='GPU for CARLA simulator')
    parser.add_argument('--spawn', type=int, default=13)
    parser.add_argument('--dest', type=int, default=34)
    parser.add_argument('--fpga-host', default='143.89.78.219')
    parser.add_argument('--fpga-port', type=int, default=12345)
    parser.add_argument('--out-dir', default='dashboard_data_fpga')
    args = parser.parse_args()
    record_eval(args)
