import sys
import carla
import random
import numpy as np
import cv2
import os
import pandas as pd
import json
import queue
import gc
import math
from tqdm import tqdm
import wandb
from carla_utils.traffic_utils import spawn_traffic, reset_vehicle
from carla_utils.route_utils import set_traffic_lights_time
# from carla_utils.visualization_utils import draw_full_diagnostic

CARLA_UE_VERSION = 4  # I think 5 isnt fesible for various reasons: doesnt support weather, no tesla model 3 that we have been using in UE4 version, and the physics are too different that the compute_smooth_control function doesnt work.


if CARLA_UE_VERSION == 4:
    sys.path.append("/path/to/carla/PythonAPI/carla") 
elif CARLA_UE_VERSION == 5:
    sys.path.append("/path/to/carla/PythonAPI/carla")
else:
    raise Exception("Unsupported CARLA version. Please set CARLA_UE_VERSION to 4 or 5.")

from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.local_planner import RoadOption
from agents.navigation.basic_agent import BasicAgent


CURRENT_SCENARIO = "DYNAMIC"

N_VEHICLES = 40
N_WALKERS = 40

OUTPUT_DIR = f"carla_ue{CARLA_UE_VERSION}_dataset_{CURRENT_SCENARIO.lower()}"
IMG_DIR = os.path.join(OUTPUT_DIR, "images")
CSV_PATH = os.path.join(OUTPUT_DIR, "data.csv")

USE_WANDB = True
WANDB_PROJECT = "vla-accel"
WANDB_SAVE_VIDEO_PROB = 0.05

TOTAL_FRAMES = 400000 
SKIP_FIRST_N_FRAMES = 120
RENDER_SIZE = (1920, 1080)
SAVE_SIZE = 256
FOV = 110
SAVE_INTERVAL = 2000 

CAM_Y_OFFSET = 0.4
CAM_YAW = 25.0
STEER_ADJUSTMENT_CAM_LR = 0.18

# Physics & Control
TARGET_SPEED = 15.0 # km/h (Slow and steady)
SPEED_BUFFER = 2.0
DROP_STRAIGHT_PROB = 0.90 # Drop 60% of straight frames to balance turns
MULTI_CAM = False

THIRD_PERSON_CAM = False

STATE_NORMAL = 0
STATE_PERTURB = 1
STATE_RECOVER = 2
timer = 0
perturb_steer = 0.0

# Seed random
random.seed(56)

def draw_full_diagnostic(img_bgr, model, cmd):
    h, w = img_bgr.shape[:2]
    
    # Semi-transparent background for UI
    overlay = img_bgr.copy()
    cv2.rectangle(overlay, (0, 0), (w, 60), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, img_bgr, 0.5, 0, img_bgr)

    # 1. Text Info
    cv2.putText(img_bgr, f"CMD: {cmd.upper()}", (10, 25), 0, 0.6, (0, 255, 255), 2)
    
    # 2. Steering Lines (Bottom)
    start_pt = (w // 2, h - 20)
    prd_steer_end = (int(w // 2 + model[0] * 150), h - 70)
    cv2.line(img_bgr, start_pt, prd_steer_end, (0, 0, 255), 2) # Model Red

    # 3. Throttle/Brake Bars (Top Right)
    # Model Bars (Red)
    # cv2.putText(img_bgr, "MODEL", (w - 180, 85), 0, 0.4, (0, 0, 255), 1)
    # Throttle
    cv2.rectangle(img_bgr, (w - 100, 35), (w - 100 + int(model[1]*80), 45), (0, 0, 255), -1)
    # Brake
    cv2.rectangle(img_bgr, (w - 100, 50), (w - 100 + int(model[2]*80), 60), (100, 0, 100), -1)

    # Labels
    cv2.putText(img_bgr, "THR", (w - 130, 40), 0, 0.3, (255, 255, 255), 1)
    cv2.putText(img_bgr, "BRK", (w - 130, 55), 0, 0.3, (255, 255, 255), 1)

    return img_bgr

def get_angle(car_transform, wp_transform):
    fwd = car_transform.get_forward_vector()
    wp_vec = wp_transform.location - car_transform.location
    v1 = np.array([fwd.x, fwd.y])
    v2 = np.array([wp_vec.x, wp_vec.y])
    v1 /= np.linalg.norm(v1)
    v2 /= np.linalg.norm(v2)
    angle = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))
    cross = v1[0] * v2[1] - v1[1] * v2[0]
    if cross < 0: angle = -angle
    return angle

def get_closest_waypoint_index(vehicle_loc, route):
    min_dist = float('inf')
    closest_idx = 0
    for i, (wp, _) in enumerate(route):
        dist = wp.transform.location.distance(vehicle_loc)
        if dist < min_dist:
            min_dist = dist
            closest_idx = i
    return closest_idx, min_dist

def capture_base_map(world, output_dir):
    """Captures a high-res top-down view of the map and saves projection data."""
    print("Capturing Base Map...")
    
    # 1. Calculate Map Bounds
    spawn_points = world.get_map().get_spawn_points()
    x_vals = [sp.location.x for sp in spawn_points]
    y_vals = [sp.location.y for sp in spawn_points]
    
    min_x, max_x = min(x_vals), max(x_vals)
    min_y, max_y = min(y_vals), max(y_vals)
    
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    
    # Add padding
    width = (max_x - min_x) * 1.2
    height = (max_y - min_y) * 1.2
    cam_z = max(width, height) # 90 deg FOV covers width=height at dist=height
    
    # 2. Spawn Camera
    cam_bp = world.get_blueprint_library().find('sensor.camera.rgb')
    cam_bp.set_attribute('image_size_x', '2048')
    cam_bp.set_attribute('image_size_y', '2048')
    cam_bp.set_attribute('fov', '90')
    
    loc = carla.Location(x=center_x, y=center_y, z=cam_z)
    rot = carla.Rotation(pitch=-90)
    camera = world.spawn_actor(cam_bp, carla.Transform(loc, rot))
    
    # 3. Capture
    q = queue.Queue()
    camera.listen(q.put)
    for _ in range(10): world.tick() # Warmup
    img = q.get()
    
    # 4. Save Image
    array = np.frombuffer(img.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (2048, 2048, 4))[:, :, :3]
    cv2.imwrite(os.path.join(output_dir, "town_map.png"), array)
    
    # 5. Save Metadata (Crucial for projection)
    metadata = {
        "cam_x": center_x, "cam_y": center_y, "cam_z": cam_z,
        "img_w": 2048, "img_h": 2048, "fov": 90
    }
    with open(os.path.join(output_dir, "map_metadata.json"), "w") as f:
        json.dump(metadata, f)
        
    camera.destroy()
    print("Base Map Saved.")

class BEVVisualizer:
    def __init__(self, world):
        self.world = world
        self.history = [] # Stores (Location, Command)
        
        # Color Map for Commands (RGB)
        self.colors = {
            "follow lane": (0, 255, 0),       # Green
            "go straight": (0, 255, 255),     # Cyan
            "turn left": (255, 255, 0),       # Yellow
            "turn right": (255, 0, 0),        # Red
            "change lane left": (255, 0, 255),# Magenta
            "change lane right": (0, 0, 255), # Blue
            "stop at red light": (255, 255, 255) # White
        }

    def record_step(self, location, command):
        # Just store the data, we draw later
        self.history.append((location, command))

    def build_projection_matrix(self, w, h, fov):
        focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
        K = np.identity(3)
        K[0, 0] = K[1, 1] = focal
        K[0, 2] = w / 2.0
        K[1, 2] = h / 2.0
        return K

    def get_image_point(self, loc, K, w2c):
        # Calculate 2D projection of 3D location
        point = np.array([loc.x, loc.y, loc.z, 1])
        # transform to camera coordinates
        point_camera = np.dot(w2c, point)

        # New we must change from UE4's coordinate system to an "standard"
        # (x, y ,z) -> (y, -z, x)
        # and we remove the fourth component also
        point_camera = [point_camera[1], -point_camera[2], point_camera[0]]

        # now project 3D->2D using the camera matrix
        if point_camera[2] < 0: return None # Behind camera

        u = K[0, 0] * point_camera[0] / point_camera[2] + K[0, 2]
        v = K[1, 1] * point_camera[1] / point_camera[2] + K[1, 2]

        return (int(u), int(v))

    def capture_bev(self, route_idx):
        """Spawns a camera, takes a pic, and draws the route using OpenCV."""
        if not self.history: return None
        
        # 1. Calculate bounds to center the camera
        locs = [h[0] for h in self.history]
        min_x = min(l.x for l in locs); max_x = max(l.x for l in locs)
        min_y = min(l.y for l in locs); max_y = max(l.y for l in locs)
        
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        width = max_x - min_x
        height = max_y - min_y
        
        # Adjust height to fit route
        cam_z = max(width, height) * 1.2
        if cam_z < 50: cam_z = 50
        
        # 2. Spawn Camera
        img_res = 1024
        fov = 90
        cam_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        cam_bp.set_attribute('image_size_x', str(img_res))
        cam_bp.set_attribute('image_size_y', str(img_res))
        cam_bp.set_attribute('fov', str(fov))
        
        loc = carla.Location(x=center_x, y=center_y, z=cam_z)
        rot = carla.Rotation(pitch=-90, yaw=0, roll=0) # Look straight down
        camera = self.world.spawn_actor(cam_bp, carla.Transform(loc, rot))
        
        # 3. Wait for image
        q = queue.Queue()
        camera.listen(q.put)
        
        # Tick to let render catch up
        for _ in range(5): self.world.tick()
        
        img_data = q.get()
        
        # 4. Convert to numpy
        array = np.frombuffer(img_data.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (img_res, img_res, 4))[:, :, :3]
        # Make writable copy for OpenCV
        array = array.copy() 
        
        # 5. PROJECT AND DRAW
        # Build matrices
        K = self.build_projection_matrix(img_res, img_res, fov)
        world_2_camera = np.array(camera.get_transform().get_inverse_matrix())
        
        for loc, cmd in self.history:
            # Project 3D location to 2D pixel
            pt = self.get_image_point(loc, K, world_2_camera)
            
            if pt:
                # Get color (RGB)
                rgb = self.colors.get(cmd, (128, 128, 128))
                # Convert to BGR for OpenCV
                bgr = (rgb[2], rgb[1], rgb[0])
                
                # Draw filled circle
                cv2.circle(array, pt, 6, bgr, -1)

        # 6. Add Legend
        y = 50
        array = array.copy()
        cv2.putText(array, "LEGEND:", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        y += 40
        
        for cmd, rgb in self.colors.items():
            bgr = (rgb[2], rgb[1], rgb[0])
            array = array.copy()
            cv2.putText(array, cmd, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, bgr, 2)
            y += 35
            
        camera.destroy()
        return array


def compute_smooth_control(vehicle, route, target_speed_kmh, hazard_detected):
    control = carla.VehicleControl()
    
    # 1. BRAKE (Binary Safety)
    if hazard_detected:
        control.brake = 1.0
        control.throttle = 0.0
        control.steer = 0.0
        return control

    # 2. STEER (Pure Pursuit)
    target_wp = None
    ego_loc = vehicle.get_location()
    for i, (wp, _) in enumerate(route):
        if wp.transform.location.distance(ego_loc) > 4.0:
            target_wp = wp
            break
    
    if target_wp:
        angle = get_angle(vehicle.get_transform(), target_wp.transform)
        control.steer = np.clip(angle * 0.6, -1.0, 1.0) 
    else:
        control.steer = 0.0

    # 3. THROTTLE (Hysteresis)
    v = vehicle.get_velocity()
    speed_kmh = 3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)
    
    if speed_kmh < (target_speed_kmh - SPEED_BUFFER):
        control.throttle = 0.50 # Accelerate
        control.brake = 0.0
    elif speed_kmh > (target_speed_kmh + SPEED_BUFFER):
        control.throttle = 0.0  # Coast
        control.brake = 0.0
    else:
        control.throttle = 0.0  # Maintain Coast in sweet spot
        control.brake = 0.0

    return control

def get_enhanced_command(route, vehicle_loc, vehicle_yaw):
    if not route: return "follow lane"
    
    # --- TUNING PARAMETERS ---
    # 1. Lane Change: Increased to 30m to catch the command before steering starts
    LANE_CHANGE_TRIGGER = 30.0 
    
    # 2. Turns: Keep at 15m (seems to work well for entry)
    TURN_TRIGGER = 15.0        
    
    # 3. Junction Entry: 10m
    JUNCTION_TRIGGER = 10.0    
    
    cumulative_dist = 0.0
    prev_loc = vehicle_loc
    
    # --- PHASE 1: SCAN AHEAD (The Plan) ---
    for i in range(min(len(route), 50)):
        wp, cmd = route[i]
        curr_loc = wp.transform.location
        cumulative_dist += curr_loc.distance(prev_loc)
        prev_loc = curr_loc
        
        # Stop scanning if we went too far
        if cumulative_dist > LANE_CHANGE_TRIGGER: break
        
        # PRIORITY 1: LANE CHANGES (Long range)
        if cmd == RoadOption.CHANGELANELEFT: return "change lane left"
        if cmd == RoadOption.CHANGELANERIGHT: return "change lane right"

        # PRIORITY 2: TURNS (Medium range)
        if cumulative_dist < TURN_TRIGGER:
            if cmd in [RoadOption.LEFT, RoadOption.RIGHT]:
                # Verify geometry (prevent noise)
                wp_yaw = wp.transform.rotation.yaw
                diff = wp_yaw - vehicle_yaw
                while diff > 180: diff -= 360
                while diff < -180: diff += 360
                
                if diff > 15: return "turn right"
                if diff < -15: return "turn left"

        # PRIORITY 3: EXPLICIT STRAIGHT (Short range)
        if cumulative_dist < JUNCTION_TRIGGER:
            if cmd == RoadOption.STRAIGHT:
                if wp.is_junction: return "go straight"

    # --- PHASE 2: GAP FILLER (The Reality) ---
    # If we are physically inside a junction, we must maintain the command
    # because the lookahead might only see "LaneFollow" for the exit.
    current_wp = route[0][0]
    
    if current_wp.is_junction:
        # Look ONLY 5 meters ahead. 
        # Reducing this from 10m -> 5m fixes the "Linger" issue.
        # We want to know what the road is doing *right now*, not in 10m.
        lookahead_dist = 0.0
        prev_loc = vehicle_loc
        
        for i in range(min(len(route), 10)):
            wp = route[i][0]
            lookahead_dist += wp.transform.location.distance(prev_loc)
            prev_loc = wp.transform.location
            
            if lookahead_dist > 5.0:
                # Calculate angle to this immediate future point
                wp_yaw = wp.transform.rotation.yaw
                diff = wp_yaw - vehicle_yaw
                while diff > 180: diff -= 360
                while diff < -180: diff += 360
                
                # Geometric Classification
                # Stricter thresholds to snap back to straight earlier
                if diff > 20: return "turn right"
                if diff < -20: return "turn left"
                
                # If angle is small (< 10), we are effectively going straight
                if abs(diff) < 10: return "go straight"
                
                # If we are between 10 and 20 degrees, we are transitioning.
                # Defaulting to "follow lane" here helps smooth the transition out of the turn.
                return "follow lane"
                
    # Default
    return "follow lane"


def check_hazards(vehicle, world, map_data):
    if vehicle.is_at_traffic_light():
        if vehicle.get_traffic_light().get_state() == carla.TrafficLightState.Red:
            return "stop at red light"

    # 2. Geometry Setup
    ego_trans = vehicle.get_transform()
    ego_loc = ego_trans.location
    ego_fwd = ego_trans.get_forward_vector()
    
    # 3. Get all relevant actors
    # (In a highly optimized simulation, you'd pass this list in, but this works)
    actors = world.get_actors()
    vehicles = actors.filter('vehicle.*')
    walkers = actors.filter('walker.pedestrian.*')
    
    # 4. Define Safety Distances
    VEHICLE_BRAKE_DIST = 8.0
    WALKER_BRAKE_DIST = 12.0
    LANE_WIDTH_TOLERANCE = 1.8 # ~Half a lane width (detects overlapping objects)

    def is_hazard(target_actor, brake_dist):
        if target_actor.id == vehicle.id: return False
        
        target_loc = target_actor.get_location()
        dist_vec = target_loc - ego_loc
        
        # Calculate Longitudinal Distance (Forward/Backward)
        # Dot product: Projects the distance vector onto the forward vector
        long_dist = dist_vec.x * ego_fwd.x + dist_vec.y * ego_fwd.y + dist_vec.z * ego_fwd.z
        
        if long_dist < 0: return False # Behind us
        if long_dist > brake_dist: return False # Too far away
        
        # Calculate Lateral Distance (Side-to-Side)
        # Using Pythagorean theorem: Lat^2 = Dist^2 - Long^2
        dist_sq = dist_vec.x**2 + dist_vec.y**2 + dist_vec.z**2
        lat_dist = math.sqrt(max(0, dist_sq - long_dist**2))
        
        # If the object is within our "Tube" width, it's a hazard
        return lat_dist < LANE_WIDTH_TOLERANCE

    # 5. Check Vehicles
    for target in vehicles:
        if is_hazard(target, VEHICLE_BRAKE_DIST):
            return "brake for vehicle"

    # 6. Check Pedestrians (New)
    for target in walkers:
        if is_hazard(target, WALKER_BRAKE_DIST):
            return "brake for pedestrian"
            
    return None


def is_lane_safe(vehicle, world, map_data, direction):
    """
    Checks if the target lane (left or right) is clear of vehicles.
    direction: 'left' or 'right'
    """
    ego_loc = vehicle.get_location()
    ego_wpt = map_data.get_waypoint(ego_loc)
    
    # 1. Get the Target Waypoint
    target_wpt = None
    if direction == 'left':
        target_wpt = ego_wpt.get_left_lane()
    else: # right
        target_wpt = ego_wpt.get_right_lane()
        
    # If no lane exists (e.g. sidewalk/wall), it's not safe
    if not target_wpt or target_wpt.lane_type != carla.LaneType.Driving:
        return False

    # 2. Define Safety Zone parameters
    # At 15km/h, we need about 8-10 meters clearance front and back
    SAFETY_DIST_FRONT = 10.0 
    SAFETY_DIST_BACK = 8.0
    LANE_WIDTH_CHECK = 1.5 # Half lane width

    # 3. Check all vehicles
    actors = world.get_actors()
    vehicles = actors.filter('vehicle.*')
    
    for target in vehicles:
        if target.id == vehicle.id: continue
        
        target_loc = target.get_location()
        
        # We check distance relative to the TARGET WAYPOINT, not our car.
        # This effectively checks "Is anyone in the spot I want to move to?"
        
        # Vector from Target Waypoint to Other Vehicle
        vec_diff = target_loc - target_wpt.transform.location
        
        # Project onto lane direction
        fwd = target_wpt.transform.get_forward_vector()
        long_dist = vec_diff.x * fwd.x + vec_diff.y * fwd.y
        
        # Check Longitudinal (Front/Back)
        if -SAFETY_DIST_BACK < long_dist < SAFETY_DIST_FRONT:
            # Check Lateral (Is it actually in that lane?)
            # Distance from the center of the target lane
            # We use 2D Euclidean distance minus the longitudinal component
            dist_sq = vec_diff.x**2 + vec_diff.y**2 + vec_diff.z**2
            lat_dist = math.sqrt(max(0, dist_sq - long_dist**2))
            
            if lat_dist < LANE_WIDTH_CHECK:
                # print(f"Lane blocked by {target.type_id} at {long_dist:.1f}m")
                return False # Unsafe

    return True

def save_chunk(data_buffer, csv_path, is_first_chunk):
    df = pd.DataFrame(data_buffer)
    df.to_csv(csv_path, mode='a', header=is_first_chunk, index=False)
    del df
    gc.collect()

def main():
    if not os.path.exists(IMG_DIR): os.makedirs(IMG_DIR)
    if os.path.exists(CSV_PATH): os.remove(CSV_PATH)

    client = carla.Client('localhost', 3000)
    client.set_timeout(20.0)
    
    print("Loading Town10HD_Opt...")
    world = client.load_world('Town10HD_Opt') 
    map_data = world.get_map()
    grp = GlobalRoutePlanner(map_data, sampling_resolution=2.0)
    
    traffic_actors = []
    is_rainy = 0
    is_crowded = 0
    if CURRENT_SCENARIO == "WEATHER":
        # Heavy Rain, Wet Roads
        weather = carla.WeatherParameters.HardRainNoon
        weather.fog_density = 10.0
        weather.wetness = 100.0
        world.set_weather(weather)
        is_rainy = 1
    elif CURRENT_SCENARIO == "DYNAMIC":
        world.set_weather(carla.WeatherParameters.ClearNoon)
        traffic_actors = spawn_traffic(world, client, N_VEHICLES, N_WALKERS)
        is_crowded = 1
    else: # BASE
        world.set_weather(carla.WeatherParameters.ClearNoon)
    
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    settings.substepping = True
    settings.max_substep_delta_time = 0.01
    settings.max_substeps = 10
    world.apply_settings(settings)

    actor_list = []
    data_buffer = []
    video_frames = []
    tp_video_frames = []
    skip_frame_count = 0

    if USE_WANDB:
        wandb.init(project=WANDB_PROJECT, name=f"dataset_debug_run_{CURRENT_SCENARIO.lower()}")

    capture_base_map(world, OUTPUT_DIR)
    expert_steer_smooth = 0.0

    try:
        bp_lib = world.get_blueprint_library()
        vehicle_bp = bp_lib.find('vehicle.tesla.model3')
        ego_spawn = random.choice(map_data.get_spawn_points())
        vehicle = world.spawn_actor(vehicle_bp, ego_spawn) 
        actor_list.append(vehicle)
        
        cam_bp = bp_lib.find('sensor.camera.rgb')
        cam_bp.set_attribute('image_size_x', str(RENDER_SIZE[0]))
        cam_bp.set_attribute('image_size_y', str(RENDER_SIZE[1]))
        cam_bp.set_attribute('fov', str(FOV))
        camera = world.spawn_actor(cam_bp, carla.Transform(carla.Location(x=1.5, z=2.4)), attach_to=vehicle)
        actor_list.append(camera)

        if THIRD_PERSON_CAM:
            view_bp = bp_lib.find('sensor.camera.rgb')
            view_bp.set_attribute('image_size_x', str(RENDER_SIZE[0]))
            view_bp.set_attribute('image_size_y', str(RENDER_SIZE[1]))
            view_bp.set_attribute('fov', str(FOV))
            view_transform = carla.Transform(
                carla.Location(x=-5.5, z=2.8), 
                carla.Rotation(pitch=-15)
            )
            tp_camera = world.spawn_actor(view_bp, view_transform, attach_to=vehicle)
            tp_queue = queue.Queue()
            tp_camera.listen(tp_queue.put)
            actor_list.append(tp_camera)

        if MULTI_CAM:
            loc_l = carla.Location(x=1.5, y=-CAM_Y_OFFSET, z=2.4)
            loc_r = carla.Location(x=1.5, y=CAM_Y_OFFSET, z=2.4)
            rot_l = carla.Rotation(yaw=-CAM_YAW) 
            rot_r = carla.Rotation(yaw=CAM_YAW)
            camera_l = world.spawn_actor(cam_bp, carla.Transform(loc_l, rot_l), attach_to=vehicle)
            camera_r = world.spawn_actor(cam_bp, carla.Transform(loc_r, rot_r), attach_to=vehicle)
            actor_list.extend([camera_l, camera_r])
        
        image_queue = queue.Queue()
        camera.listen(image_queue.put)
        if MULTI_CAM:
            queue_l = queue.Queue()
            camera_l.listen(queue_l.put)
            queue_r = queue.Queue()
            camera_r.listen(queue_r.put)

        current_route = []
        pbar = tqdm(total=TOTAL_FRAMES)
        frame_count = 0
        trajectory_id = 0 
        is_first_chunk = True
        settling_timer = 0

        def clear_q(q):
            while not q.empty():
                try: q.get_nowait()
                except queue.Empty: break

        clear_q(image_queue)
        if THIRD_PERSON_CAM:
            clear_q(tp_queue)
        if MULTI_CAM:
            clear_q(queue_l)
            clear_q(queue_r)
        
        # C. Warmup: Skip next few ticks to let the car fall and stabilize
        for _ in range(30): 
            world.tick()
            try:
                image_queue.get(block=True, timeout=1.0)
                if THIRD_PERSON_CAM:
                    tp_queue.get(block=True, timeout=1.0)
                if MULTI_CAM:
                    queue_l.get(block=True, timeout=1.0)
                    queue_r.get(block=True, timeout=1.0)
            except queue.Empty:
                print("[!] Warning: Sensor timeout during warmup reset.")
      
        world.tick()
        # Create collision sensor
        col_bp = bp_lib.find('sensor.other.collision')
        col_sensor = world.spawn_actor(col_bp, carla.Transform(), attach_to=vehicle)
        col_queue = queue.Queue()
        col_sensor.listen(col_queue.put)
        actor_list.append(col_sensor)

        set_traffic_lights_time(world)

        # Variable to track how long we've been stopped
        stuck_timer = 0

        state = STATE_NORMAL
        perturb_timer = 0
        dont_perturb_after_recovery_for = 80
        wandb_save_video = random.random() < WANDB_SAVE_VIDEO_PROB

        while frame_count < TOTAL_FRAMES:
            
            ego_loc = vehicle.get_location()
            ego_yaw = vehicle.get_transform().rotation.yaw
            # Collision check
            collided = False
            if not col_queue.empty():
                while not col_queue.empty(): col_queue.get() # clear queue
                print("\n[!] Collision Detected! Resetting...")
                collided = True

            # --- 2. DETECT STUCK ---
            v = vehicle.get_velocity()
            speed = 3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)
            
            # Check if we are stopped but NOT because of a hazard (red light/vehicle)
            is_stopped = speed < 1.0
            hazard_msg = check_hazards(vehicle, world, map_data)
            
            if is_stopped and (hazard_msg is None):
                stuck_timer += 1
            else:
                stuck_timer = 0
                
            is_stuck = stuck_timer > 50 # Stuck for 5 seconds (at 10 FPS)

            if collided or is_stuck:
                # A. Cleanup Data: Remove the last 80 frames (the "approach" to the crash)
                # This prevents the model from learning the actions that led to the failure.
                n_to_remove = min(len(data_buffer), 50)
                if n_to_remove > 0:
                    print(f"Cleaning dataset: removing last {n_to_remove} frames.")
                    data_buffer = data_buffer[:-n_to_remove]
                    # Decrement pbar/frame_count if you want to be precise
                    pbar.update(-n_to_remove)
                    frame_count -= n_to_remove

                # B. Reset Vehicle
                reset_vehicle(vehicle, map_data)
                current_route = [] # Force route recalculation
                state = STATE_NORMAL # Reset perturbation state machine
                stuck_timer = 0
                trajectory_id += 1 

                clear_q(image_queue)
                if THIRD_PERSON_CAM:
                    clear_q(tp_queue)
                if MULTI_CAM:
                    clear_q(queue_l)
                    clear_q(queue_r)
                
                # C. Warmup: Skip next few ticks to let the car fall and stabilize
                for _ in range(30): 
                    world.tick()
                    try:
                        image_queue.get(block=True, timeout=1.0)
                        if THIRD_PERSON_CAM:
                            tp_queue.get(block=True, timeout=1.0)
                        if MULTI_CAM:
                            queue_l.get(block=True, timeout=1.0)
                            queue_r.get(block=True, timeout=1.0)
                    except queue.Empty:
                        print("[!] Warning: Sensor timeout during warmup reset.")
                world.tick()
                continue # Skip the rest of this loop iteration
            # Finish Collision Check

            try:
                carla_image = image_queue.get(block=True, timeout=2.0)
                if THIRD_PERSON_CAM:
                    carla_image_tp = tp_queue.get(block=True, timeout=2.0)
                if MULTI_CAM:
                    carla_image_l = queue_l.get(block=True, timeout=2.0)
                    carla_image_r = queue_r.get(block=True, timeout=2.0)
            except queue.Empty: 
                trajectory_id += 1 
                continue

            # Skip frames during settling (e.g. after reset)
            if settling_timer > 0:
                settling_timer -= 1
                continue

            # wp = map_data.get_waypoint(ego_loc, project_to_road=True)

            # --- 1. ROUTE MANAGEMENT ---
            # Check if current route is less than 10 and we are not in the middle of a junction
            if len(current_route) < 10: # and not wp.is_junction:
                spawn_points = map_data.get_spawn_points()
                dest = random.choice(spawn_points).location
                current_route = grp.trace_route(ego_loc, dest)
                # Trim start
                min_d = 1000
                start_idx = 0
                for i, (wp, cmd) in enumerate(current_route):
                    d = wp.transform.location.distance(ego_loc)
                    if d < min_d:
                        min_d = d
                        start_idx = i
                current_route = current_route[start_idx:]

            # Prune local route
            while len(current_route) > 0:
                wp, cmd = current_route[0]
                if wp.transform.location.distance(ego_loc) < 4.0:
                    current_route.pop(0)
                else:
                    break

            hazard_msg = check_hazards(vehicle, world, map_data)
            control = None

            v = vehicle.get_velocity()
            speed_m_s = math.sqrt(v.x**2 + v.y**2 + v.z**2)

            perturb_prob = 0.007
            if CURRENT_SCENARIO == "DYNAMIC":
                perturb_prob = 0.002

            if state == STATE_NORMAL:
                control = compute_smooth_control(vehicle, current_route, TARGET_SPEED, hazard_detected=(hazard_msg is not None))
                control.manual_gear_shift = False
                vehicle.apply_control(control)
                perturb_timer += 1
                if random.random() < perturb_prob and hazard_msg is None and perturb_timer > dont_perturb_after_recovery_for:
                    state = STATE_PERTURB
                    timer = random.randint(15, 25)
                    perturb_steer = random.choice([-0.20, 0.20])
                    perturb_timer = 0
                    wpt_start = map_data.get_waypoint(vehicle.get_location(), project_to_road=True)
                    start_loc = wpt_start.transform.location
                    perturb_start_dist = math.sqrt((ego_loc.x - start_loc.x)**2 + (ego_loc.y - start_loc.y)**2)

                record_this_frame = True
            elif state == STATE_PERTURB:
                drift_control = carla.VehicleControl(
                    steer=perturb_steer, 
                    throttle=0.14, # 0.3,
                    brake=0.0,
                    manual_gear_shift=False
                )
                waypoint_now = map_data.get_waypoint(ego_loc, project_to_road=True)
                target_loc = waypoint_now.transform.location
                current_dist = math.sqrt((ego_loc.x - target_loc.x)**2 + (ego_loc.y - target_loc.y)**2)

                if abs(current_dist - perturb_start_dist) > 0.8:
                    state = STATE_RECOVER
                    control = compute_smooth_control(vehicle, current_route, TARGET_SPEED, hazard_detected=(hazard_msg is not None))
                    timer = 15
                else:
                    vehicle.apply_control(drift_control)
                    timer -= 1
                    if timer <= 0:
                        state = STATE_RECOVER
                        timer = 15 # Give it 3 seconds to recover
                        control = compute_smooth_control(vehicle, current_route, TARGET_SPEED, hazard_detected=(hazard_msg is not None))
                record_this_frame = False
            elif state == STATE_RECOVER:
                control = compute_smooth_control(vehicle, current_route, TARGET_SPEED, hazard_detected=(hazard_msg is not None))
                control.manual_gear_shift = False
                vehicle.apply_control(control)
                record_this_frame = False
                
                timer -= 1
                waypoint_now = map_data.get_waypoint(ego_loc, project_to_road=True)
                target_loc = waypoint_now.transform.location
                dist_to_centerline = math.sqrt((ego_loc.x - target_loc.x)**2 + (ego_loc.y - target_loc.y)**2)
                if timer <= 0 or dist_to_centerline < 0.3:
                    state = STATE_NORMAL
                    trajectory_id += 1 

            world.tick()

            ego_loc = vehicle.get_location()
            ego_yaw = vehicle.get_transform().rotation.yaw

            if skip_frame_count < SKIP_FIRST_N_FRAMES:
                skip_frame_count += 1
                continue

            if hazard_msg: instruction = hazard_msg
            else: instruction = get_enhanced_command(current_route, ego_loc, ego_yaw)

            _, dist_to_center = get_closest_waypoint_index(ego_loc, current_route)
            is_mid_maneuver = dist_to_center > 0.5
            if not is_mid_maneuver:
                if instruction == "change lane left":
                    if not is_lane_safe(vehicle, world, map_data, 'left'):
                        # BLOCKED: Downgrade to 'Follow Lane'
                        # The car will drive straight and wait for the gap to open.
                        instruction = "follow lane"
                        # print("Holding Lane Change Left (Unsafe)")
                        
                elif instruction == "change lane right":
                    if not is_lane_safe(vehicle, world, map_data, 'right'):
                        # BLOCKED: Downgrade to 'Follow Lane'
                        instruction = "follow lane"
                        # print("Holding Lane Change Right (Unsafe)")


            if not record_this_frame:
                '''  # For debugging the perturbation/recovery phases
                array_tp = np.frombuffer(carla_image_tp.raw_data, dtype=np.dtype("uint8"))
                array_tp = np.reshape(array_tp, (RENDER_SIZE[1], RENDER_SIZE[0], 4))[:, :, :3]
                wandb_tp_array = array_tp.copy()
                if state == STATE_PERTURB:
                    text_color = (0, 255, 255) # Yellow for perturbation
                    action = (perturb_steer, 0.4, 0)
                elif state == STATE_RECOVER:
                    text_color = (0, 255, 0) # Green for recovery
                    action = (control.steer, control.throttle, control.brake)
                else:
                    text_color = (255, 255, 255) # White for normal
                    action = (control.steer, control.throttle, control.brake)
                wandb_tp_array = draw_full_diagnostic(wandb_tp_array, action, instruction)
                str_tp_view = "NORMAL" if state == STATE_NORMAL else ("PERTURB" if state == STATE_PERTURB else "RECOVER")
                wandb_tp_array = cv2.putText(wandb_tp_array, str_tp_view, (240, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, text_color, 2)
                wandb_tp_array = cv2.cvtColor(wandb_tp_array, cv2.COLOR_BGR2RGB)
                wandb_tp_array = np.transpose(wandb_tp_array, (2, 0, 1))
                tp_video_frames.append(wandb_tp_array)
                '''
                continue
 
            # Balancing Logic
            is_boring = (
                abs(control.steer) < 0.05 and 
                instruction == "follow lane" and
                state == STATE_NORMAL
            )
            
            is_train_candidate = True
            if is_boring and random.random() < DROP_STRAIGHT_PROB:
                is_train_candidate = False

            # Super-Sampling Resize
            array = np.frombuffer(carla_image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (RENDER_SIZE[1], RENDER_SIZE[0], 4))[:, :, :3]

            if THIRD_PERSON_CAM:
                array_tp = np.frombuffer(carla_image_tp.raw_data, dtype=np.dtype("uint8"))
                array_tp = np.reshape(array_tp, (RENDER_SIZE[1], RENDER_SIZE[0], 4))[:, :, :3]

            if MULTI_CAM:
                array_l = np.frombuffer(carla_image_l.raw_data, dtype=np.dtype("uint8"))
                array_r = np.frombuffer(carla_image_r.raw_data, dtype=np.dtype("uint8"))
                array_l = np.reshape(array_l, (RENDER_SIZE[1], RENDER_SIZE[0], 4))[:, :, :3]
                array_r = np.reshape(array_r, (RENDER_SIZE[1], RENDER_SIZE[0], 4))[:, :, :3]

            if USE_WANDB:
                # wandb_array = cv2.resize(array, (RENDER_SIZE[0], RENDER_SIZE[1]), interpolation=cv2.INTER_AREA)
                wandb_array = array.copy()
                wandb_tp_array = array_tp.copy() if THIRD_PERSON_CAM else None
                # wandb_array_l = array_l.copy()
                # wandb_array_r = array_r.copy()
            # Center crop to square
            '''
            ah, aw = array.shape[:2]
            side = min(ah, aw)
            start_x = (aw - side) // 2
            start_y = (ah - side) // 2
            array = array[start_y:start_y+side, start_x:start_x+side]
            array = array.copy()
            '''
            array = cv2.resize(array, (SAVE_SIZE, SAVE_SIZE), interpolation=cv2.INTER_AREA)

            if THIRD_PERSON_CAM:
                array_tp = array_tp[start_y:start_y+side, start_x:start_x+side]
                array_tp = cv2.resize(array_tp, (SAVE_SIZE, SAVE_SIZE), interpolation=cv2.INTER_AREA)

            if MULTI_CAM:
                array_l = array_l[start_y:start_y+side, start_x:start_x+side]
                array_l = cv2.resize(array_l, (SAVE_SIZE, SAVE_SIZE), interpolation=cv2.INTER_AREA)
                array_r = array_r[start_y:start_y+side, start_x:start_x+side]
                array_r = cv2.resize(array_r, (SAVE_SIZE, SAVE_SIZE), interpolation=cv2.INTER_AREA)

            filename = f"{frame_count:06d}"
            cv2.imwrite(os.path.join(IMG_DIR, filename + "_c.png"), array)
            if MULTI_CAM:
                cv2.imwrite(os.path.join(IMG_DIR, filename + "_l.png"), array_l)
                cv2.imwrite(os.path.join(IMG_DIR, filename + "_r.png"), array_r)
            
            if USE_WANDB:
                if wandb_save_video:
                    wandb_array = draw_full_diagnostic(wandb_array, (control.steer, control.throttle, control.brake), instruction)
                    wandb_array = cv2.cvtColor(wandb_array, cv2.COLOR_BGR2RGB)
                    wandb_array = np.transpose(wandb_array, (2, 0, 1))
                    video_frames.append(wandb_array)

                if THIRD_PERSON_CAM:
                    wandb_tp_array = draw_full_diagnostic(wandb_tp_array, (control.steer, control.throttle, control.brake), instruction)
                    str_tp_view = "NORMAL" if state == STATE_NORMAL else ("PERTURB" if state == STATE_PERTURB else "RECOVER")
                    wandb_tp_array = cv2.putText(wandb_tp_array, str_tp_view, (240, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                    wandb_tp_array = cv2.cvtColor(wandb_tp_array, cv2.COLOR_BGR2RGB)
                    wandb_tp_array = np.transpose(wandb_tp_array, (2, 0, 1))
                    tp_video_frames.append(wandb_tp_array)
          
            record = {
                "image_path": filename,
                "instruction": instruction, 
                "steer": control.steer,
                "throttle": control.throttle,
                "brake": control.brake,
                "speed": speed_m_s,
                "location_x": ego_loc.x,
                "location_y": ego_loc.y,
                "is_train_candidate": int(is_train_candidate),
                "trajectory_id": trajectory_id,
                "scenario": CURRENT_SCENARIO,
                "is_rainy": is_rainy,
                "is_crowded": is_crowded
            }
            if MULTI_CAM:
                record_l = record.copy()
                record_r = record.copy()
            record["image_path"] = filename + "_c.png"
            data_buffer.append(record)

            if MULTI_CAM:
                steer_l = np.clip(control.steer + STEER_ADJUSTMENT_CAM_LR, -1.0, 1.0)
                steer_r = np.clip(control.steer - STEER_ADJUSTMENT_CAM_LR, -1.0, 1.0)
                record_l["steer"] = steer_l
                record_l["image_path"] = filename + "_l.png"
                data_buffer.append(record_l)
                record_r["steer"] = steer_r
                record_r["image_path"] = filename + "_r.png"
                data_buffer.append(record_r)
            
            # bev_viz.record_step(ego_loc, instruction)
            frame_count += 1
            pbar.update(1)

            db_len_limit = SAVE_INTERVAL * 3 if MULTI_CAM else SAVE_INTERVAL
            
            if len(data_buffer) >= db_len_limit:
                save_chunk(data_buffer, CSV_PATH, is_first_chunk)
                data_buffer = []
                is_first_chunk = False
                if USE_WANDB:
                    if wandb_save_video:
                        video = np.stack(video_frames)
                        wandb.log({f"Trajectory {trajectory_id} Video": wandb.Video(video, fps=20, format="mp4")})
                    video_frames = []
                    wandb_save_video = random.random() < WANDB_SAVE_VIDEO_PROB
                    if THIRD_PERSON_CAM:
                        tp_video = np.stack(tp_video_frames)
                        wandb.log({f"Trajectory {trajectory_id} TP Video": wandb.Video(tp_video, fps=20, format="mp4")})
                        tp_video_frames = []
            
    finally:
        if len(data_buffer) > 0:
            save_chunk(data_buffer, CSV_PATH, is_first_chunk)
        if traffic_actors is not None:
            print("Cleaning up traffic actors...")
            client.apply_batch([carla.command.DestroyActor(x) for x in traffic_actors])
        for actor in actor_list: 
            if actor.is_alive: actor.destroy()

if __name__ == "__main__":
    main()
