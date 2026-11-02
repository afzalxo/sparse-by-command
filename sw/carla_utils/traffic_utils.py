import carla
import random
import socket


def find_free_port(starting_port):
    port = starting_port
    while True:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("localhost", port))
                return port
        except OSError:
            port += 1


def reset_vehicle(vehicle, map_data):
    """Teleports vehicle to a random valid spawn point and clears velocity."""
    spawn_points = map_data.get_spawn_points()
    spawn_point = random.choice(spawn_points)
    vehicle.set_transform(spawn_point)
    vehicle.set_target_velocity(carla.Vector3D(0, 0, 0))
    vehicle.set_target_angular_velocity(carla.Vector3D(0, 0, 0))
    return spawn_point


def spawn_traffic(world, client, n_vehicles, n_walkers, gpu_rank=0):
    """Spawns NPCs and Walkers for the Dynamic Scenario."""
    print(f"Spawning Traffic: {n_vehicles} Vehicles, {n_walkers} Walkers...")
    actor_list = []
    vehicle_list = []
    walker_list = []

    # 1. Setup Traffic Manager
    # Each gpu_rank starts scanning from a different 100-port block so parallel
    # subprocesses never collide, even if a previous TM is still lingering.
    tm_port = find_free_port(8000 + gpu_rank * 100)
    print(f"Using Traffic Manager on port {tm_port} (gpu_rank={gpu_rank})")
    tm = client.get_trafficmanager(tm_port)  #8000)
    tm.set_random_device_seed(42)
    tm.set_synchronous_mode(True)
    tm.set_hybrid_physics_mode(True) 
    tm.global_percentage_speed_difference(30.0)
    tm.set_global_distance_to_leading_vehicle(3)

    # 2. Spawn Vehicles
    bp_lib = world.get_blueprint_library()
    vehicle_bps = bp_lib.filter('vehicle.*')
    vehicle_bps = [x for x in vehicle_bps if int(x.get_attribute('number_of_wheels')) == 4]
    spawn_points = world.get_map().get_spawn_points()
    spawn_points.pop(86)
    spawn_points.pop(81) # TODO: Remove our spawn point from the list to avoid collision
    spawn_points.pop(72)
    spawn_points.pop(66)
    spawn_points.pop(31)
    spawn_points.pop(13)
    
    # Randomly shuffle spawn points
    random.shuffle(spawn_points)
    
    for i in range(min(n_vehicles, len(spawn_points))):
        bp = random.choice(vehicle_bps)
        if bp.has_attribute('color'):
            color = random.choice(bp.get_attribute('color').recommended_values)
            bp.set_attribute('color', color)
        
        # Set autopilot
        bp.set_attribute('role_name', 'autopilot')
        spawn_point = spawn_points[i]
        spawn_point.location.z += 0.3
        
        try:
            v = world.spawn_actor(bp, spawn_point)
            v.set_autopilot(True, tm.get_port())
            tm.ignore_lights_percentage(v, 0) # Obey lights
            tm.auto_lane_change(v, True)
            actor_list.append(v)
            vehicle_list.append(v)
        except:
            pass
            
    # 3. Spawn Walkers
    # Walkers need a controller + the actual walker actor
    walker_bp = bp_lib.filter("walker.pedestrian.*")
    controller_bp = bp_lib.find('controller.ai.walker')
    
    spawn_points = []
    for _ in range(n_walkers):
        loc = world.get_random_location_from_navigation()
        if loc: spawn_points.append(carla.Transform(loc))
    
    for spawn_point in spawn_points:
        try:
            bp = random.choice(walker_bp)
            walker = world.spawn_actor(bp, spawn_point)
            controller = world.spawn_actor(controller_bp, carla.Transform(), attach_to=walker)
            
            # Start walking
            controller.start()
            controller.go_to_location(world.get_random_location_from_navigation())
            controller.set_max_speed(1.4 + random.random()) # Random speed
            
            actor_list.append(walker)
            actor_list.append(controller)
            walker_list.append(walker)
        except:
            pass

    print(f"Spawned {len(vehicle_list)} Vehicles and {len(walker_list)} Walkers.")
    return actor_list


