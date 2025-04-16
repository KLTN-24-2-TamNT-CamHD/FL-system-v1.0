import json
import logging
import os
import time
from datetime import datetime
from typing import Dict, Any, List, Optional, Set
import uuid

class TrainingEvent:
    """Base class for training events."""
    
    def __init__(self, event_type: str):
        self.event_type = event_type
        self.timestamp = time.time()
        self.event_id = str(uuid.uuid4())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "timestamp": self.timestamp,
            "timestamp_iso": datetime.fromtimestamp(self.timestamp).isoformat()
        }
    
    def to_json(self) -> str:
        """Convert event to JSON string."""
        return json.dumps(self.to_dict())


class ClientParticipationEvent(TrainingEvent):
    """Event for client participation in training."""
    
    def __init__(
        self, 
        client_id: str, 
        client_address: str, 
        round_id: int,
        properties: Optional[Dict[str, Any]] = None
    ):
        super().__init__("client_participation")
        self.client_id = client_id
        self.client_address = client_address
        self.round_id = round_id
        self.properties = properties or {}
    
    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            "client_id": self.client_id,
            "client_address": self.client_address,
            "round_id": self.round_id,
            "properties": self.properties
        })
        return data


class ClientUpdateEvent(TrainingEvent):
    """Event for client model update submission."""
    
    def __init__(
        self, 
        client_id: str, 
        client_address: str,
        round_id: int,
        ipfs_hash: str,
        tx_hash: Optional[str] = None,
        metrics: Optional[Dict[str, Any]] = None
    ):
        super().__init__("client_update")
        self.client_id = client_id
        self.client_address = client_address
        self.round_id = round_id
        self.ipfs_hash = ipfs_hash
        self.tx_hash = tx_hash
        self.metrics = metrics or {}
    
    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            "client_id": self.client_id,
            "client_address": self.client_address,
            "round_id": self.round_id,
            "ipfs_hash": self.ipfs_hash,
            "tx_hash": self.tx_hash,
            "metrics": self.metrics
        })
        return data


class TrainingRoundEvent(TrainingEvent):
    """Event for training round lifecycle."""
    
    def __init__(
        self,
        round_id: int,
        round_state: str,  # "started", "aggregated", "completed"
        global_model_hash: Optional[str] = None,
        tx_hash: Optional[str] = None,
        metrics: Optional[Dict[str, Any]] = None,
        participating_clients: Optional[List[str]] = None
    ):
        super().__init__("training_round")
        self.round_id = round_id
        self.round_state = round_state
        self.global_model_hash = global_model_hash
        self.tx_hash = tx_hash
        self.metrics = metrics or {}
        self.participating_clients = participating_clients or []
    
    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            "round_id": self.round_id,
            "round_state": self.round_state,
            "global_model_hash": self.global_model_hash,
            "tx_hash": self.tx_hash,
            "metrics": self.metrics,
            "participating_clients": self.participating_clients
        })
        return data


class ServerLogger:
    """Logger for federated learning server events."""
    
    def __init__(
        self, 
        log_dir: str, 
        experiment_name: Optional[str] = None,
        experiment_id: Optional[str] = None,
        console_log_level: int = logging.INFO,
        file_log_level: int = logging.DEBUG
    ):
        """
        Initialize the server logger.
        
        Args:
            log_dir: Directory for log files
            experiment_name: Name of the experiment (used in filenames)
            experiment_id: Optional experiment ID
            console_log_level: Logging level for console output
            file_log_level: Logging level for file output
        """
        self.log_dir = log_dir
        self.experiment_name = experiment_name or f"fl_experiment_{int(time.time())}"
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Setup Python logging
        self.logger = logging.getLogger("federated.server")
        self.logger.setLevel(logging.DEBUG)
        
        # Clear existing handlers to avoid duplicates
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(console_log_level)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler
        log_file = os.path.join(log_dir, f"{self.experiment_name}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(file_log_level)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # Setup JSON event log files
        self.events_file = os.path.join(log_dir, f"{self.experiment_name}_events.jsonl")
        self.metrics_file = os.path.join(log_dir, f"{self.experiment_name}_metrics.json")
        
        # In-memory state
        self.active_clients: Dict[str, Dict[str, Any]] = {}  # client_id -> info
        self.client_addresses: Dict[str, str] = {}  # client_id -> blockchain address
        self.rounds: Dict[int, Dict[str, Any]] = {}  # round_id -> info
        self.current_round_id: Optional[int] = None
        self.transactions: List[Dict[str, Any]] = []  # List of tx details
        
        self.logger.info(f"ServerLogger initialized for experiment: {self.experiment_name}")
        self.logger.info(f"Log files will be stored in: {log_dir}")
    
    def log_event(self, event: TrainingEvent) -> None:
        """
        Log a training event to the events file.
        
        Args:
            event: Training event to log
        """
        with open(self.events_file, "a") as f:
            f.write(event.to_json() + "\n")
        
        self.logger.debug(f"Logged event: {event.event_type} ({event.event_id})")
    
    def log_client_participation(
        self, 
        client_id: str, 
        client_address: str, 
        round_id: int,
        properties: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log client participation in training.
        
        Args:
            client_id: Unique identifier for the client
            client_address: Blockchain address of the client
            round_id: Current training round ID
            properties: Additional client properties
        """
        event = ClientParticipationEvent(
            client_id=client_id,
            client_address=client_address,
            round_id=round_id,
            properties=properties
        )
        
        self.log_event(event)
        
        # Update in-memory state
        self.active_clients[client_id] = {
            "client_id": client_id,
            "client_address": client_address,
            "last_seen": time.time(),
            "properties": properties or {}
        }
        
        self.client_addresses[client_id] = client_address
        
        # Ensure round exists
        if round_id not in self.rounds:
            self.rounds[round_id] = {
                "round_id": round_id,
                "participating_clients": set(),
                "completed_clients": set(),
                "start_time": time.time()
            }
        
        # Add client to participants
        self.rounds[round_id]["participating_clients"].add(client_id)
        
        self.logger.info(f"Client {client_id} (address: {client_address}) joined round {round_id}")
    
    def log_client_update(
        self,
        client_id: str,
        round_id: int,
        ipfs_hash: str,
        tx_hash: Optional[str] = None,
        metrics: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log client model update submission.
        
        Args:
            client_id: Unique identifier for the client
            round_id: Current training round ID
            ipfs_hash: IPFS hash of the model parameters
            tx_hash: Blockchain transaction hash
            metrics: Client training metrics
        """
        # Get client address from memory
        client_address = self.client_addresses.get(client_id, "unknown")
        
        event = ClientUpdateEvent(
            client_id=client_id,
            client_address=client_address,
            round_id=round_id,
            ipfs_hash=ipfs_hash,
            tx_hash=tx_hash,
            metrics=metrics
        )
        
        self.log_event(event)
        
        # Update in-memory state
        if round_id in self.rounds:
            # Mark client as completed
            self.rounds[round_id]["completed_clients"].add(client_id)
            
            # Store update info
            if "updates" not in self.rounds[round_id]:
                self.rounds[round_id]["updates"] = {}
                
            self.rounds[round_id]["updates"][client_id] = {
                "ipfs_hash": ipfs_hash,
                "tx_hash": tx_hash,
                "metrics": metrics,
                "timestamp": time.time()
            }
        
        # Store transaction info if available
        if tx_hash:
            self.transactions.append({
                "tx_hash": tx_hash,
                "client_id": client_id,
                "round_id": round_id,
                "ipfs_hash": ipfs_hash,
                "event_type": "client_update",
                "timestamp": time.time()
            })
            
        self.logger.info(f"Client {client_id} submitted update for round {round_id}: {ipfs_hash}")
        if metrics:
            self.logger.debug(f"Client {client_id} metrics: {metrics}")
    
    def log_round_start(self, round_id: int, tx_hash: Optional[str] = None) -> None:
        """
        Log the start of a training round.
        
        Args:
            round_id: Training round ID
            tx_hash: Blockchain transaction hash
        """
        event = TrainingRoundEvent(
            round_id=round_id,
            round_state="started",
            tx_hash=tx_hash
        )
        
        self.log_event(event)
        
        # Update in-memory state
        self.current_round_id = round_id
        self.rounds[round_id] = {
            "round_id": round_id,
            "state": "started",
            "start_time": time.time(),
            "participating_clients": set(),
            "completed_clients": set(),
            "tx_hash": tx_hash
        }
        
        # Store transaction info if available
        if tx_hash:
            self.transactions.append({
                "tx_hash": tx_hash,
                "round_id": round_id,
                "event_type": "round_start",
                "timestamp": time.time()
            })
            
        self.logger.info(f"Training round {round_id} started")
    
    def log_round_aggregation(
        self, 
        round_id: int, 
        global_model_hash: str,
        metrics: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log the aggregation of a training round.
        
        Args:
            round_id: Training round ID
            global_model_hash: IPFS hash of the aggregated global model
            metrics: Aggregation metrics
        """
        # Get participating clients
        participating_clients = []
        if round_id in self.rounds:
            participating_clients = list(self.rounds[round_id]["participating_clients"])
        
        event = TrainingRoundEvent(
            round_id=round_id,
            round_state="aggregated",
            global_model_hash=global_model_hash,
            metrics=metrics,
            participating_clients=participating_clients
        )
        
        self.log_event(event)
        
        # Update in-memory state
        if round_id in self.rounds:
            self.rounds[round_id]["state"] = "aggregated"
            self.rounds[round_id]["aggregation_time"] = time.time()
            self.rounds[round_id]["global_model_hash"] = global_model_hash
            self.rounds[round_id]["metrics"] = metrics or {}
        
        self.logger.info(f"Training round {round_id} aggregated: {global_model_hash}")
    
    def log_round_completion(
        self, 
        round_id: int, 
        global_model_hash: str,
        tx_hash: Optional[str] = None,
        metrics: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log the completion of a training round.
        
        Args:
            round_id: Training round ID
            global_model_hash: IPFS hash of the final global model
            tx_hash: Blockchain transaction hash
            metrics: Final round metrics
        """
        # Get participating clients
        participating_clients = []
        if round_id in self.rounds:
            participating_clients = list(self.rounds[round_id]["participating_clients"])
        
        event = TrainingRoundEvent(
            round_id=round_id,
            round_state="completed",
            global_model_hash=global_model_hash,
            tx_hash=tx_hash,
            metrics=metrics,
            participating_clients=participating_clients
        )
        
        self.log_event(event)
        
        # Update in-memory state
        if round_id in self.rounds:
            self.rounds[round_id]["state"] = "completed"
            self.rounds[round_id]["completion_time"] = time.time()
            self.rounds[round_id]["global_model_hash"] = global_model_hash
            self.rounds[round_id]["tx_hash"] = tx_hash
            self.rounds[round_id]["metrics"] = metrics or {}
        
        # Store transaction info if available
        if tx_hash:
            self.transactions.append({
                "tx_hash": tx_hash,
                "round_id": round_id,
                "global_model_hash": global_model_hash,
                "event_type": "round_completion",
                "timestamp": time.time()
            })
            
        self.logger.info(f"Training round {round_id} completed: {global_model_hash}")
    
    def save_metrics(self) -> None:
        """Save all metrics to the metrics file."""
        metrics = {
            "experiment": self.experiment_name,
            "timestamp": time.time(),
            "rounds": {str(k): self._set_to_list(v) for k, v in self.rounds.items()},
            "clients": self.active_clients,
            "transactions": self.transactions
        }
        
        with open(self.metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)
            
        self.logger.info(f"Metrics saved to {self.metrics_file}")
    
    def _set_to_list(self, d: Dict[str, Any]) -> Dict[str, Any]:
        """Convert sets in a dictionary to lists for JSON serialization."""
        result = {}
        for k, v in d.items():
            if isinstance(v, set):
                result[k] = list(v)
            elif isinstance(v, dict):
                result[k] = self._set_to_list(v)
            else:
                result[k] = v
        return result
    
    def get_round_stats(self, round_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Get statistics for a specific round or the current round.
        
        Args:
            round_id: Round ID or None for current round
            
        Returns:
            Dictionary with round statistics
        """
        if round_id is None:
            round_id = self.current_round_id
            
        if round_id is None or round_id not in self.rounds:
            return {}
            
        round_data = self.rounds[round_id]
        
        # Convert sets to lists for easier handling
        result = self._set_to_list(round_data)
        
        # Add some calculated stats
        if "start_time" in round_data:
            result["duration"] = time.time() - round_data["start_time"]
            
        result["total_clients"] = len(round_data["participating_clients"])
        result["completed_clients"] = len(round_data["completed_clients"])
        
        if result["total_clients"] > 0:
            result["completion_rate"] = result["completed_clients"] / result["total_clients"]
        else:
            result["completion_rate"] = 0
            
        return result