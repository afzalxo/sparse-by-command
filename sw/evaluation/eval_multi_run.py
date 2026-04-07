"""Evaluate a checkpoint N times on all 3 hard routes to get mean±std."""
import sys, os, time, random, multiprocessing, json
import numpy as np
import torch
import carla

sys.path.append("/path/to/carla/PythonAPI/carla")
from carla_utils.eval_utils import evaluate_in_carla_gated, evaluate_in_carla
from carla_utils.carla_server import CarlaServer
from carla_utils.traffic_utils import spawn_traffic
from models.carla_model import CarlaVLAHW_Gated as CarlaVLA

ROUTES = [(31, 99), (13, 34), (72, 106)]
ROUTE_NAMES = {(31, 99): "Hard 1", (13, 34): "Hard 2", (72, 106): "Hard 3"}
N_RUNS = 5
GPUS = [4, 5, 6, 7, 8]  # 5 GPUs for 5 parallel runs
BASE_PORT = 25000

def run_full_eval(run_id, model_path, gpu_id, rpc_port, hard_masking, result_queue, dense_model=False):
    """Run eval on all 3 routes sequentially in one CARLA instance."""
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    device = torch.device(f"cuda:{gpu_id}")
    run_results = {}
    
    print(f"[Run {run_id}] Starting CARLA on GPU {gpu_id}, port {rpc_port}", flush=True)
    carla_server = CarlaServer(gpu_rank=gpu_id, rpc_port=rpc_port)
    carla_proc = carla_server.start()
    
    if carla_proc is None:
        print(f"[Run {run_id}] CARLA failed to start!", flush=True)
        result_queue.put((run_id, None))
        return
    
    try:
        client = carla.Client('localhost', rpc_port)
        client.set_timeout(60.0)
        
        if dense_model:
            from models.carla_model import CarlaVLAHW_Dense
            model = CarlaVLAHW_Dense().to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
        else:
            model = CarlaVLA(task_dim=6).to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        total_route_covered = 0.0
        total_route_length = 0.0
        total_cte = 0.0
        total_frames = 0
        
        for route in ROUTES:
            # Reload world for each route (clean state)
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
            
            for _ in range(10):
                world.tick()
            
            traffic_actors = spawn_traffic(world, client, n_vehicles=40, n_walkers=40, gpu_rank=gpu_id)
            
            if dense_model:
                route_covered, route_length, cte, frame_count = evaluate_in_carla(
                    model, 0, device, rpc_port, world, "dynamic",
                    spawn_index=route[0], dest_index=route[1],
                    eval_render_size=(1920, 1080), fov=110,
                    save_to_wandb=False
                )
            else:
                route_covered, route_length, cte, frame_count = evaluate_in_carla_gated(
                    model, 0, device, rpc_port, world, "dynamic",
                    spawn_index=route[0], dest_index=route[1],
                    eval_render_size=(1920, 1080), fov=110,
                    save_to_wandb=False, hard_masking=hard_masking
                )
            
            completion = (route_covered / route_length) * 100.0 if route_length > 0 else 0.0
            avg_cte = cte / frame_count if frame_count > 0 else 0.0
            
            route_name = ROUTE_NAMES[route]
            run_results[route_name] = {"completion": completion, "cte": avg_cte, "frames": frame_count}
            print(f"[Run {run_id}] {route_name} ({route[0]}->{route[1]}): Completion={completion:.2f}% CTE={avg_cte:.3f}m Frames={frame_count}", flush=True)
            
            total_route_covered += route_covered
            total_route_length += route_length
            total_cte += cte
            total_frames += frame_count
            
            # Clean up traffic
            client.apply_batch([carla.command.DestroyActor(x) for x in traffic_actors])
            time.sleep(2)
        
        overall_completion = (total_route_covered / total_route_length) * 100.0 if total_route_length > 0 else 0.0
        overall_cte = total_cte / total_frames if total_frames > 0 else 0.0
        run_results["overall"] = {"completion": overall_completion, "cte": overall_cte, "frames": total_frames}
        print(f"[Run {run_id}] OVERALL: Completion={overall_completion:.2f}% CTE={overall_cte:.3f}m", flush=True)
        
        result_queue.put((run_id, run_results))
        
    except Exception as e:
        import traceback
        print(f"[Run {run_id}] ERROR: {e}", flush=True)
        traceback.print_exc()
        result_queue.put((run_id, None))
    finally:
        carla_server.stop()
        time.sleep(5)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--hard-masking', action='store_true')
    parser.add_argument("--dense", action="store_true", help="Use dense model")
    parser.add_argument('--n-runs', type=int, default=N_RUNS)
    parser.add_argument('--gpus', type=int, nargs='+', default=GPUS)
    args = parser.parse_args()
    
    n_runs = args.n_runs
    gpus = args.gpus
    
    print(f"=== Multi-Run Eval: {n_runs} runs on GPUs {gpus} ===", flush=True)
    print(f"Model: {args.model_path}", flush=True)
    print(f"Hard masking: {args.hard_masking}", flush=True)
    print(f"Routes: {ROUTES}", flush=True)
    
    # Verify GPUs are clean
    for gpu_id in gpus[:n_runs]:
        os.system(f"pkill -9 -f 'graphicsadapter={gpu_id}' 2>/dev/null")
    time.sleep(5)
    
    ctx = multiprocessing.get_context('spawn')
    queues = []
    processes = []
    
    for i in range(n_runs):
        gpu_id = gpus[i % len(gpus)]
        rpc_port = BASE_PORT + i * 100
        q = ctx.Queue()
        queues.append(q)
        p = ctx.Process(target=run_full_eval, 
                       args=(i+1, args.model_path, gpu_id, rpc_port, args.hard_masking, q, args.dense))
        processes.append(p)
    
    # Start all in parallel
    for p in processes:
        p.start()
        time.sleep(5)  # Stagger CARLA startups
    
    # Wait for all
    for p in processes:
        p.join(timeout=600)  # 10 min timeout
    
    # Collect results
    all_results = []
    for q in queues:
        try:
            result = q.get(timeout=10)
            all_results.append(result)
        except:
            all_results.append((0, None))
    
    # Print summary
    print(f"\n{'='*70}", flush=True)
    print("=== MULTI-RUN EVALUATION RESULTS ===", flush=True)
    print(f"{'='*70}", flush=True)
    
    valid_results = [(rid, r) for rid, r in all_results if r is not None]
    
    for run_id, results in valid_results:
        ovr = results["overall"]
        print(f"  Run {run_id}: Completion={ovr['completion']:.2f}%  CTE={ovr['cte']:.3f}m", flush=True)
    
    if len(valid_results) >= 2:
        # Per-route stats
        for route_name in ["Hard 1", "Hard 2", "Hard 3", "overall"]:
            completions = [r[route_name]["completion"] for _, r in valid_results]
            ctes = [r[route_name]["cte"] for _, r in valid_results]
            print(f"\n  {route_name}:", flush=True)
            print(f"    Completion: {np.mean(completions):.2f}% ± {np.std(completions):.2f}%", flush=True)
            print(f"    CTE: {np.mean(ctes):.3f}m ± {np.std(ctes):.3f}m", flush=True)
    
    # Save to JSON
    output = {
        "model_path": args.model_path,
        "hard_masking": args.hard_masking,
        "n_runs": n_runs,
        "routes": [str(r) for r in ROUTES],
        "runs": {str(rid): r for rid, r in valid_results},
    }
    if len(valid_results) >= 2:
        stats = {}
        for route_name in ["Hard 1", "Hard 2", "Hard 3", "overall"]:
            completions = [r[route_name]["completion"] for _, r in valid_results]
            ctes = [r[route_name]["cte"] for _, r in valid_results]
            stats[route_name] = {
                "completion_mean": float(np.mean(completions)),
                "completion_std": float(np.std(completions)),
                "cte_mean": float(np.mean(ctes)),
                "cte_std": float(np.std(ctes)),
            }
        output["stats"] = stats
    
    json_path = args.model_path.replace(".pth", "_eval_results.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {json_path}", flush=True)
