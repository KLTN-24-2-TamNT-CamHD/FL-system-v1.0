import asyncio
import logging
import subprocess
import os
import signal
import sys
import time
from typing import Dict, List, Any, Optional, Tuple
import json
from datetime import datetime
import pathlib
import threading

# Setup logging
logger = logging.getLogger(__name__)

class FlowerServerMonitor:
    """
    Dedicated class to monitor Flower server in a background thread
    """
    def __init__(self, connector):
        self.connector = connector
        self.running = False
        self.thread = None
    
    def start(self):
        """Start the monitor in a background thread"""
        if self.running:
            return
            
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
    
    def stop(self):
        """Stop the monitor"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
            self.thread = None
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.running:
            # Only proceed if training is in progress
            if not self.connector.training_in_progress:
                time.sleep(1)
                continue
                
            # Check if server process is running
            if self.connector.server_process and self.connector.server_process.poll() is None:
                # Process is running - update status from logs
                self._update_status_from_logs()
            else:
                # Process has ended
                if self.connector.server_process and self.connector.server_process.returncode == 0:
                    # Successful completion
                    self.connector.current_round = self.connector.total_rounds
                    logger.info(f"Flower server completed successfully")
                else:
                    # Error or manual termination
                    logger.error(f"Flower server exited with code: {self.connector.server_process.returncode if self.connector.server_process else 'unknown'}")
                
                self.connector.training_in_progress = False
                
            # Sleep between checks to avoid high CPU usage
            time.sleep(5)
    
    def _update_status_from_logs(self):
        """Update server status by reading log file"""
        import re
        
        if not os.path.exists(self.connector.server_log_file):
            return
            
        try:
            with open(self.connector.server_log_file, "r") as log_file:
                lines = log_file.readlines()
                
                # Process new lines (only lines we haven't processed yet)
                start_line = self.connector.last_processed_line if hasattr(self.connector, 'last_processed_line') else 0
                new_lines = lines[start_line:]
                
                if not new_lines:
                    return
                    
                # Update the last processed line
                self.connector.last_processed_line = len(lines)
                
                # Print new lines to terminal
                for line in new_lines:
                    line = line.strip()
                    if line:
                        # Log to console
                        if "ERROR" in line or "CRITICAL" in line:
                            logger.error(f"Flower Server: {line}")
                        elif "WARNING" in line:
                            logger.warning(f"Flower Server: {line}")
                        elif "INFO" in line:
                            logger.info(f"Flower Server: {line}")
                        elif "DEBUG" in line:
                            logger.debug(f"Flower Server: {line}")
                        else:
                            # Regular output
                            print(f"Flower Server: {line}")
                
                # Analyze the last 100 lines for training progress updates
                for line in lines[-100:]:
                    # Pattern to detect round start/progress
                    if "FL round" in line or "round " in line:
                        try:
                            # Find digits after "round" keyword
                            round_matches = re.findall(r'round (\d+)', line.lower())
                            if round_matches:
                                potential_round = int(round_matches[0])
                                if potential_round > self.connector.current_round:
                                    self.connector.current_round = potential_round
                                    logger.info(f"Training progress: Round {self.connector.current_round}/{self.connector.total_rounds}")
                        except (ValueError, IndexError):
                            pass  # Ignore parsing errors
                    
                    # Look for client connections
                    if "client" in line.lower() and ("connect" in line.lower() or "joined" in line.lower()):
                        try:
                            # Try to extract client ID from the log line
                            client_id_match = re.search(r'client[_\s]*(\w+)', line.lower())
                            if client_id_match:
                                client_id = client_id_match.group(1)
                                self.connector.client_statuses[f"client_{client_id}"] = "active"
                                logger.info(f"Detected client {client_id} as active")
                        except Exception:
                            pass  # Ignore parsing errors
                
                # Check for completion signals
                for line in lines[-20:]:  # Check recent lines
                    if (("completed" in line.lower() and "round" in line.lower()) or
                        "server completed" in line.lower() or 
                        "all metrics saved" in line.lower()):
                        # Training might be complete
                        if self.connector.current_round >= self.connector.total_rounds - 1:  # -1 to account for zero-indexing
                            self.connector.current_round = self.connector.total_rounds
                            logger.info("Training completed based on logs")
                            
        except Exception as e:
            logger.error(f"Error reading log file: {str(e)}")

class FlowerServerConnector:
    """
    Connector class to interact with the Flower server for federated learning
    """
    
    def __init__(self, server_address: str = "localhost:8088"):
        self.server_address = server_address
        self.training_in_progress = False
        self.current_round = 0
        self.total_rounds = 0
        self.start_time = None
        self.client_statuses = {}
        self.server_process = None
        self.server_log_file = "flower_server.log"
        
        # Determine the path to the Flower server script
        # Find the v2_fl directory (sibling to v2_api)
        current_dir = pathlib.Path(__file__).parent.absolute()  # core directory
        api_root = current_dir.parent.parent  # v2_api directory
        fl_dir = api_root.parent / "v2_fl"  # v2_fl directory
        
        self.server_script_path = str(fl_dir / "server.py")
        logger.info(f"Flower server script path: {self.server_script_path}")
        
        # Verify the server script exists
        if not os.path.exists(self.server_script_path):
            logger.warning(f"Flower server script not found at {self.server_script_path}")
            # Fallback to environment variable if script not found
            self.server_script_path = os.getenv("FLOWER_SERVER_SCRIPT", "server.py")
            logger.info(f"Using fallback server script path: {self.server_script_path}")
        
        # Create monitor
        self.monitor = None
        
    async def initialize(self) -> bool:
        """
        Initialize the Flower server connector
        """
        logger.info(f"Initializing Flower server connector at {self.server_address}")
        return True
    
    async def start_server(self, config: Dict[str, Any]) -> bool:
        """
        Start the Flower server process
        """
        if self.server_process and self.server_process.poll() is None:
            logger.warning("Flower server is already running")
            return False
            
        try:
            # Prepare command with arguments from config
            cmd = [
                sys.executable,
                self.server_script_path,
                "--server-address", self.server_address,
                "--rounds", str(config.get("num_rounds", 3)),
                "--min-fit-clients", str(config.get("min_fit_clients", 2)),
                "--min-evaluate-clients", str(config.get("min_evaluate_clients", 2)),
                "--fraction-fit", str(config.get("fraction_fit", 1.0))
            ]
            
            # Add other config parameters for your specific server
            if "ipfs_url" in config and config["ipfs_url"]:
                cmd.extend(["--ipfs-url", config["ipfs_url"]])
            else:
                cmd.extend(["--ipfs-url", "http://127.0.0.1:5001/api/v0"])
                
            if "ganache_url" in config and config["ganache_url"]:
                cmd.extend(["--ganache-url", config["ganache_url"]])
            else:
                cmd.extend(["--ganache-url", "http://192.168.1.146:7545"])
                
            if "contract_address" in config and config["contract_address"]:
                cmd.extend(["--contract-address", config["contract_address"]])
                
            if "private_key" in config and config["private_key"]:
                cmd.extend(["--private-key", config["private_key"]])
                
            if config.get("deploy_contract", False):
                cmd.append("--deploy-contract")
                
            if "version_prefix" in config and config["version_prefix"]:
                cmd.extend(["--version-prefix", config["version_prefix"]])
            else:
                cmd.extend(["--version-prefix", "1.0"])
                
            if config.get("authorized_clients_only", False):
                cmd.append("--authorized-clients-only")
                
            if "authorized_clients" in config and config["authorized_clients"]:
                # Filter out any None values
                valid_clients = [client for client in config["authorized_clients"] if client is not None]
                if valid_clients:
                    cmd.extend(["--authorize-clients"] + valid_clients)
                
            if "round_rewards" in config and config["round_rewards"] is not None:
                cmd.extend(["--round-rewards", str(config["round_rewards"])])
            else:
                cmd.extend(["--round-rewards", "1000"])
                
            if "device" in config and config["device"]:
                cmd.extend(["--device", config["device"]])
            else:
                cmd.extend(["--device", "cpu"])
            
            # Start server process
            logger.info(f"Starting Flower server with command: {' '.join(cmd)}")
            
            # Ensure the log directory exists
            log_dir = os.path.dirname(self.server_log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)
                
            # Open log file for writing
            log_file = open(self.server_log_file, "w")
            
            # Create a pipe for real-time logging
            self.server_process = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                # Use the v2_fl directory as working directory
                cwd=os.path.dirname(self.server_script_path)
            )
            
            # Reset log processing
            self.last_processed_line = 0
            
            # Setup training monitoring
            self.training_in_progress = True
            self.current_round = 0
            self.total_rounds = config.get("num_rounds", 3)
            self.start_time = datetime.now()
            
            # Start a separate thread for monitoring
            if not self.monitor:
                self.monitor = FlowerServerMonitor(self)
            self.monitor.start()
            
            logger.info(f"Flower server started successfully (PID: {self.server_process.pid})")
            return True
            
        except Exception as e:
            logger.error(f"Error starting Flower server: {str(e)}")
            return False
    
    async def read_server_logs(self, num_lines: int = 100) -> List[str]:
        """
        Read the most recent lines from the server log file
        """
        if not os.path.exists(self.server_log_file):
            return ["No log file found"]
            
        try:
            with open(self.server_log_file, "r") as log_file:
                lines = log_file.readlines()
                return lines[-num_lines:] if len(lines) > num_lines else lines
                
        except Exception as e:
            logger.error(f"Error reading server logs: {str(e)}")
            return [f"Error reading logs: {str(e)}"]
    
    async def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the Flower server
        """
        status = {
            "server_running": False,
            "current_round": 0,
            "total_rounds": self.total_rounds,
            "started_at": None,
            "active_clients": 0
        }
        
        # Check if server process is running
        if self.server_process and self.server_process.poll() is None:
            status["server_running"] = True
        elif self.training_in_progress:
            # The process finished but training is still marked as in progress
            # This can happen with quick blockchain operations
            status["server_running"] = True
        
        # Add training progress information
        status["current_round"] = self.current_round
        
        # Add timing information
        if self.start_time:
            status["started_at"] = self.start_time.strftime("%Y-%m-%dT%H:%M:%S")
        
        # Count active clients
        active_clients = [client_id for client_id, client_status in self.client_statuses.items() 
                        if client_status == "active"]
        status["active_clients"] = len(active_clients)
        
        return status
                
    async def stop_training(self) -> bool:
        """
        Stop the current training session
        """
        if not self.server_process:
            logger.warning("No server process to stop")
            return False
            
        try:
            # Stop the monitor first if it exists
            if self.monitor:
                self.monitor.stop()
                
            # Send SIGTERM signal to gracefully stop the server
            self.server_process.terminate()
            
            # Wait for up to 5 seconds for the process to terminate
            for _ in range(10):
                if self.server_process.poll() is not None:
                    break
                await asyncio.sleep(0.5)
                
            # If still running, force kill
            if self.server_process.poll() is None:
                self.server_process.kill()
                logger.warning("Had to force kill the server process")
                
            # Update status
            self.training_in_progress = False
            
            return True
            
        except Exception as e:
            logger.error(f"Error stopping server: {str(e)}")
            return False