import subprocess
import os
import psutil
from datetime import datetime
from typing import Optional, Dict, List
import json
from app.services import blockchain_service

class FlowerServerService:
    def __init__(self):
        self.server_process: Optional[subprocess.Popen] = None
        self.server_pid: Optional[int] = None
        self.start_time: Optional[str] = None
        self.timestamp = "2025-03-31 16:07:03"
        self.admin = "dinhcam89"
        
        # Path to store server status
        self.status_file = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'data',
            'fl_server_status.json'
        )
        
        # Create data directory if it doesn't exist
        os.makedirs(os.path.dirname(self.status_file), exist_ok=True)
        
        # Load previous status if exists
        self._load_status()
        
        print(f"FL Service initialized at {self.timestamp}")
        print(f"Admin: {self.admin}")
        
    def start_server(self, port: int = 8080) -> Dict:
        """Start the Flower server if it's not already running"""
        if self.is_server_running():
            return {
                "message": "Flower server is already running",
                "status": "running",
                "pid": self.server_pid,
                "start_time": self.start_time,
                "timestamp": self.timestamp,
                "admin": self.admin,
                "blockchain_connected": blockchain_service.is_connected()
            }
        
        try:
            # Start FL server script
            server_script = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "fl_server",
                "server.py"
            )
            
            # Create log directory if it doesn't exist
            log_dir = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                'data',
                'logs'
            )
            os.makedirs(log_dir, exist_ok=True)
            
            # Open log files
            stdout_file = open(os.path.join(log_dir, 'fl_server_stdout.log'), 'a')
            stderr_file = open(os.path.join(log_dir, 'fl_server_stderr.log'), 'a')
            
            # Start server process
            self.server_process = subprocess.Popen(
                ["python", server_script, str(port)],
                stdout=stdout_file,
                stderr=stderr_file,
                start_new_session=True  # This ensures the server keeps running if the API restarts
            )
            
            self.server_pid = self.server_process.pid
            self.start_time = self.timestamp
            
            # Save status
            self._save_status()
            
            return {
                "message": "Flower server started successfully",
                "status": "running",
                "pid": self.server_pid,
                "port": port,
                "start_time": self.start_time,
                "timestamp": self.timestamp,
                "admin": self.admin,
                "blockchain_connected": blockchain_service.is_connected()
            }
            
        except Exception as e:
            return {
                "message": f"Failed to start Flower server: {str(e)}",
                "status": "failed",
                "timestamp": self.timestamp,
                "admin": self.admin,
                "blockchain_connected": blockchain_service.is_connected()
            }

    def stop_server(self) -> Dict:
        """Stop the Flower server if it's running"""
        if not self.is_server_running():
            return {
                "message": "No Flower server is running",
                "status": "stopped",
                "timestamp": self.timestamp,
                "admin": self.admin,
                "blockchain_connected": blockchain_service.is_connected()
            }
        
        try:
            # Kill the process and its children
            parent = psutil.Process(self.server_pid)
            children = parent.children(recursive=True)
            
            # Kill children first
            for child in children:
                child.terminate()
                try:
                    child.wait(timeout=5)  # Wait up to 5 seconds
                except psutil.TimeoutExpired:
                    child.kill()  # Force kill if not terminated
            
            # Kill parent
            parent.terminate()
            try:
                parent.wait(timeout=5)
            except psutil.TimeoutExpired:
                parent.kill()
            
            self.server_pid = None
            self.start_time = None
            self._save_status()
            
            return {
                "message": "Flower server stopped successfully",
                "status": "stopped",
                "timestamp": self.timestamp,
                "admin": self.admin,
                "blockchain_connected": blockchain_service.is_connected()
            }
        except Exception as e:
            return {
                "message": f"Failed to stop Flower server: {str(e)}",
                "status": "error",
                "timestamp": self.timestamp,
                "admin": self.admin,
                "blockchain_connected": blockchain_service.is_connected()
            }

    def get_status(self) -> Dict:
        """Get current server status"""
        is_running = self.is_server_running()
        blockchain_connected = blockchain_service.is_connected()
        
        status = {
            "status": "running" if is_running else "stopped",
            "pid": self.server_pid if is_running else None,
            "start_time": self.start_time if is_running else None,
            "timestamp": self.timestamp,
            "admin": self.admin,
            "blockchain_connected": blockchain_connected
        }
        
        # Add blockchain info if connected
        if blockchain_connected and is_running:
            try:
                current_round = blockchain_service.get_current_round()
                status["current_round"] = current_round
            except:
                status["current_round"] = -1
        
        # Add server logs if running
        if is_running:
            status["logs"] = self._get_recent_logs()
            
        return status

    def is_server_running(self) -> bool:
        """Check if the Flower server is running"""
        if self.server_pid is None:
            return False
            
        try:
            process = psutil.Process(self.server_pid)
            return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
        except psutil.NoSuchProcess:
            self.server_pid = None
            self.start_time = None
            self._save_status()
            return False

    def _save_status(self):
        """Save server status to file"""
        status = {
            "pid": self.server_pid,
            "start_time": self.start_time,
            "timestamp": self.timestamp,
            "admin": self.admin
        }
        
        with open(self.status_file, 'w') as f:
            json.dump(status, f)

    def _load_status(self):
        """Load server status from file"""
        try:
            with open(self.status_file, 'r') as f:
                status = json.load(f)
                self.server_pid = status.get('pid')
                self.start_time = status.get('start_time')
                
                # Verify if the loaded process is actually running
                if not self.is_server_running():
                    self.server_pid = None
                    self.start_time = None
                    self._save_status()
        except FileNotFoundError:
            pass

    def _get_recent_logs(self, lines: int = 100) -> Dict[str, List[str]]:
        """Get recent server logs"""
        logs = {"stdout": [], "stderr": []}
        log_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'data',
            'logs'
        )
        
        # Read stdout logs
        stdout_file = os.path.join(log_dir, 'fl_server_stdout.log')
        if os.path.exists(stdout_file):
            with open(stdout_file, 'r') as f:
                logs["stdout"] = f.readlines()[-lines:]
        
        # Read stderr logs
        stderr_file = os.path.join(log_dir, 'fl_server_stderr.log')
        if os.path.exists(stderr_file):
            with open(stderr_file, 'r') as f:
                logs["stderr"] = f.readlines()[-lines:]
        
        return logs

    def get_client_list(self) -> List[Dict]:
        """Get list of connected clients"""
        if not self.is_server_running():
            return []
            
        try:
            # This is a placeholder - implement actual client tracking
            return []
        except Exception:
            return []

# Initialize FL service
fl_service = FlowerServerService()