import subprocess
import os
import signal
import time
import socket


class CarlaServer:
    def __init__(self, carla_path="/home/eeafzal/carla_simulator/CarlaUE4.sh", gpu_rank=0, rpc_port=3000):
        self.carla_path = carla_path
        self.rpc_port = rpc_port
        self.proc = None
        self.gpu_rank = gpu_rank

    def start(self):
        """Starts the CARLA server and waits for it to be ready."""
        if self.is_server_running(self.rpc_port):
            print("CARLA server is already running.")
            return
        
        cmd = [
            self.carla_path,
            "-RenderOffScreen",
            "-nosound",
            f"-graphicsadapter={self.gpu_rank}",
            f"-carla-rpc-port={self.rpc_port}",
            "-quality-level=Epic"
        ]
        
        print(f"Starting CARLA Server on RPC port {self.rpc_port} GPU {self.gpu_rank}...")
        # Start process in a new process group so we can kill it easily later
        self.proc = subprocess.Popen(cmd, preexec_fn=os.setsid)
        
        # Wait for the server to accept connections
        timeout = 60
        start_time = time.time()
        while not self.is_server_running(self.rpc_port):
            if time.time() - start_time > timeout:
                print("Timeout waiting for CARLA to start!")
                return None
            time.sleep(1)
        print("CARLA Server is running.")
        return self.proc

    def stop(self):
        """Kills the CARLA server process tree."""
        if self.proc:
            print("Shutting down CARLA Server...")
            os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
            # time.sleep(5) # Give OS time to free up the port

    def is_server_running(self, port=3000):
        """Checks if the CARLA port is open."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', port)) == 0

