import carla
import math
import sys
sys.path.append("/home/eeafzal/carla_simulator/PythonAPI/carla")
from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.local_planner import RoadOption


def command_to_branch_id(cmd):
    cmd = str(cmd).lower().strip()
    if "left" in cmd:
        if "change" in cmd: return 3 # Change Lane Left
        return 1 # Turn Left
    elif "right" in cmd:
        if "change" in cmd: return 4 # Change Lane Right
        return 2 # Turn Right
    elif "stop" in cmd or "red light" in cmd or "brake" in cmd:
        return 5 # Stop at Red Light, Brake for Vehicle/Walker
    else:
        # Follow lane, go straight
        return 0

def get_enhanced_command(route, vehicle_loc, vehicle_yaw):
    if not route: return "follow lane"
    
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

def get_closest_waypoint_index(vehicle_loc, route):
    min_dist = float('inf')
    closest_idx = 0
    for i, (wp, _) in enumerate(route):
        dist = wp.transform.location.distance(vehicle_loc)
        if dist < min_dist:
            min_dist = dist
            closest_idx = i
    return closest_idx, min_dist

def calculate_route_length(route):
    length = 0.0
    for i in range(len(route) - 1):
        length += route[i][0].transform.location.distance(route[i+1][0].transform.location)
    return length


def set_traffic_lights_time(world):
    actors = world.get_actors()
    for actor in actors:
        if isinstance(actor, carla.TrafficLight):
            actor.set_state(carla.TrafficLightState.Green)
            actor.set_green_time(10)
            actor.set_yellow_time(1)
            actor.set_red_time(2)
