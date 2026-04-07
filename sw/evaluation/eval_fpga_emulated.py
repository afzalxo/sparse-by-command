"""Evaluate INT8-emulated FPGA path on 3 hard routes."""
import sys, os, time, math, queue
import numpy as np
import torch
import carla
import cv2

sys.path.append("/path/to/carla/PythonAPI/carla")
from agents.navigation.global_route_planner import GlobalRoutePlanner
from carla_utils.carla_server import CarlaServer
from carla_utils.traffic_utils import spawn_traffic
from carla_utils.route_utils import get_closest_waypoint_index, check_hazards, is_lane_safe, get_enhanced_command, command_to_branch_id
from models.carla_model import CarlaVLAHW_Gated as CarlaVLA
from fpga_emulator import FPGAEmulator

MODEL_PATH = "model_checkpoints_sparse/carla_vla_furthest_driver_phase3_gated.pth"
DATA_DIR = "fpga_data_fresh"
ROUTES = [(31, 99), (13, 34), (72, 106)]
ROUTE_NAMES = {(31, 99): "Hard 1", (13, 34): "Hard 2", (72, 106): "Hard 3"}

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--gpu-id", type=int, default=2)
    args = p.parse_args()
    gpu_id = args.gpu_id

    import random
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device(f"cuda:{gpu_id}")

    print(f"Loading model and FPGA emulator on GPU {gpu_id}...", flush=True)
    model = CarlaVLA(task_dim=6).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    emu = FPGAEmulator(DATA_DIR, model, device=device)
    print("Emulator ready.", flush=True)

    carla_server = CarlaServer(gpu_rank=gpu_id, rpc_port=30000 + gpu_id * 1000)
    carla_proc = carla_server.start()
    if carla_proc is None:
        print("CARLA failed!")
        return

    client = carla.Client('localhost', 30000 + gpu_id * 1000)
    client.set_timeout(60.0)

    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

    total_rc, total_rl, total_cte, total_fc = 0.0, 0.0, 0.0, 0

    for route in ROUTES:
        spawn_idx, dest_idx = route
        rname = ROUTE_NAMES[route]

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
        for _ in range(10): world.tick()

        traffic_actors = spawn_traffic(world, client, n_vehicles=40, n_walkers=0, gpu_rank=gpu_id)

        map_data = world.get_map()
        grp = GlobalRoutePlanner(map_data, sampling_resolution=2.0)
        spawn_points = map_data.get_spawn_points()
        start_pose = spawn_points[spawn_idx]
        dest_pose = spawn_points[dest_idx]

        full_route = grp.trace_route(start_pose.location, dest_pose.location)
        total_route_length = 0.0
        cumulative = [0.0]
        for i in range(len(full_route) - 1):
            l1 = full_route[i][0].transform.location
            l2 = full_route[i+1][0].transform.location
            d = math.sqrt((l1.x - l2.x)**2 + (l1.y - l2.y)**2)
            total_route_length += d
            cumulative.append(total_route_length)

        bp = world.get_blueprint_library().find('vehicle.tesla.model3')
        vehicle = world.spawn_actor(bp, start_pose)
        cam_bp = world.get_blueprint_library().find('sensor.camera.rgb')
        cam_bp.set_attribute('image_size_x', '1920')
        cam_bp.set_attribute('image_size_y', '1080')
        cam_bp.set_attribute('fov', '110')
        camera = world.spawn_actor(cam_bp, carla.Transform(carla.Location(x=1.5, z=2.4)), attach_to=vehicle)
        col_bp = world.get_blueprint_library().find('sensor.other.collision')
        collision_sensor = world.spawn_actor(col_bp, carla.Transform(), attach_to=vehicle)

        img_q = queue.Queue()
        camera.listen(img_q.put)
        col_q = queue.Queue()
        collision_sensor.listen(col_q.put)

        for _ in range(30):
            world.tick()
            try: img_q.get(block=False)
            except: pass
        vehicle.apply_control(carla.VehicleControl(manual_gear_shift=True, gear=1))
        for _ in range(5): world.tick()
        vehicle.apply_control(carla.VehicleControl(throttle=1.0))
        for _ in range(10): world.tick()
        vehicle.apply_control(carla.VehicleControl(throttle=0.4, manual_gear_shift=False))
        while not img_q.empty(): img_q.get_nowait()
        for _ in range(50):
            world.tick()
            try: img_q.get(block=False)
            except: pass
        world.tick()

        prev_action = torch.zeros(1, 3).to(device)
        current_route = list(full_route)
        max_reached = 0
        route_cte = 0.0
        route_frames = 0
        stop_counter = 0

        print(f"\nEvaluating {rname} ({spawn_idx}->{dest_idx})...", flush=True)

        for f in range(5000):
            ego_loc = vehicle.get_location()
            ego_yaw = vehicle.get_transform().rotation.yaw

            if not col_q.empty():
                print("Collision!", flush=True)
                break
            carla_img = img_q.get(block=True, timeout=2.0)
            _, dist_to_route = get_closest_waypoint_index(ego_loc, full_route)
            route_cte += dist_to_route
            route_frames += 1

            ego_wp = map_data.get_waypoint(ego_loc, project_to_road=False, lane_type=carla.LaneType.Any)
            if ego_wp and ego_wp.lane_type in [carla.LaneType.Sidewalk]:
                print("Off-road!", flush=True)
                break
            if dist_to_route > 20.0:
                print(f"CTE too high ({dist_to_route:.1f}m)", flush=True)
                break
            if ego_loc.distance(dest_pose.location) < 5.0:
                print("Success!", flush=True)
                break

            v_vec = vehicle.get_velocity()
            speed_ms = math.sqrt(v_vec.x**2 + v_vec.y**2 + v_vec.z**2)
            if speed_ms < 0.5 and check_hazards(vehicle, world, map_data) is None:
                stop_counter += 1
                if stop_counter > 100:
                    print("Stuck!", flush=True)
                    break
            else:
                stop_counter = 0

            arr = np.frombuffer(carla_img.raw_data, dtype=np.uint8)
            arr = np.reshape(arr, (1080, 1920, 4))[:, :, :3]
            res = cv2.resize(arr, (256, 256), interpolation=cv2.INTER_AREA)
            resized = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
            t = torch.from_numpy(resized).permute(2, 0, 1).float().to(device) / 255.0
            input_stack = (t.unsqueeze(0) - mean) / std

            sl = min(len(current_route), 20)
            ci, md = 0, float('inf')
            for i in range(sl):
                d = current_route[i][0].transform.location.distance(ego_loc)
                if d < md: md = d; ci = i
            if md < 15.0: current_route = current_route[ci:]

            hazard = check_hazards(vehicle, world, map_data)
            instr = hazard if hazard else get_enhanced_command(current_route, ego_loc, ego_yaw)
            _, dtc = get_closest_waypoint_index(ego_loc, current_route)
            if dtc <= 0.5:
                if instr == "change lane left" and not is_lane_safe(vehicle, world, map_data, 'left'):
                    instr = "follow lane"
                elif instr == "change lane right" and not is_lane_safe(vehicle, world, map_data, 'right'):
                    instr = "follow lane"
            branch = command_to_branch_id(instr)

            # ----- FPGA EMULATED FORWARD -----
            s, accel = emu.forward(input_stack, prev_action, branch)

            if accel > 0: throttle, brake = accel, 0.0
            else: throttle, brake = 0.0, -accel
            if brake < 0.1: brake = 0.0
            if speed_ms < 0.1 and brake < 0.1 and "stop" not in instr:
                throttle = max(throttle, 0.3)

            vehicle.apply_control(carla.VehicleControl(steer=float(s), throttle=float(throttle), brake=float(brake)))
            world.tick()

            prev_action = torch.tensor([s, accel, speed_ms]).unsqueeze(0).to(device)

            ego_loc = vehicle.get_location()
            se = min(max_reached + 20, len(full_route))
            best_dist, best_idx = float('inf'), max_reached
            for i in range(max_reached, se):
                wpl = full_route[i][0].transform.location
                d = math.sqrt((ego_loc.x - wpl.x)**2 + (ego_loc.y - wpl.y)**2)
                if d < best_dist and d < 5.0:
                    best_dist = d; best_idx = i
            if best_idx > max_reached: max_reached = best_idx

        comp = (cumulative[max_reached] / total_route_length) * 100.0
        acte = route_cte / route_frames if route_frames > 0 else 0.0
        print(f"FPGA-EMU Result {rname} ({spawn_idx}->{dest_idx}): Completion={comp:.2f}% | CTE={acte:.3f}m", flush=True)

        total_rc += cumulative[max_reached]
        total_rl += total_route_length
        total_cte += route_cte
        total_fc += route_frames

        try:
            tm = client.get_trafficmanager()
            tm.shut_down()
        except: pass
        client.apply_batch([carla.command.DestroyActor(x) for x in traffic_actors])
        time.sleep(2)

    oc = (total_rc / total_rl) * 100.0 if total_rl > 0 else 0.0
    ocet = total_cte / total_fc if total_fc > 0 else 0.0
    print(f"\nFPGA-EMU Overall: Completion={oc:.2f}% | CTE={ocet:.3f}m", flush=True)

    carla_server.stop()

if __name__ == "__main__":
    main()
