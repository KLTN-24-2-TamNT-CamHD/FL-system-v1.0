from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime

class TrainingConfig(BaseModel):
    """Training configuration for the federated learning system with GA-Stacking"""
    num_rounds: int = 3  # Number of federated learning rounds
    min_fit_clients: int = 2  # Minimum number of clients for training
    min_evaluate_clients: int = 2  # Minimum number of clients for evaluation
    fraction_fit: float = 1.0  # Fraction of clients to use for training
    ipfs_url: str = "http://127.0.0.1:5001/api/v0"  # IPFS API URL
    ganache_url: str = "http://192.168.1.146:7545"  # Ganache blockchain URL
    contract_address: Optional[str] = None  # Federation contract address
    private_key: Optional[str] = None  # Private key for blockchain transactions
    deploy_contract: bool = False  # Whether to deploy a new contract
    version_prefix: str = "1.0"  # Version prefix for model versioning
    authorized_clients_only: bool = False  # Whether to only accept contributions from authorized clients
    authorized_clients: Optional[List[str]] = None  # List of client addresses to authorize
    round_rewards: int = 1000  # Reward points to distribute each round
    device: str = "cpu"  # Device to use for computation
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "num_rounds": 3,
                    "min_fit_clients": 2,
                    "min_evaluate_clients": 2,
                    "fraction_fit": 1.0,
                    "ipfs_url": "http://127.0.0.1:5001/api/v0",
                    "ganache_url": "http://192.168.1.146:7545",
                    "contract_address": "0x123abc...",
                    "version_prefix": "1.0",
                    "round_rewards": 1000,
                    "device": "cpu"
                }
            ]
        }
    }

class TrainingStatus(BaseModel):
    """Status of the current training process"""
    status: str
    server_running: bool
    current_round: int
    total_rounds: int
    started_at: Optional[str] = None
    active_clients: int
    client_statuses: Dict[str, str]

class TrainingResponse(BaseModel):
    """Response for training operations"""
    status: str
    message: str
    training_id: Optional[str] = None

class ClientMetrics(BaseModel):
    """Metrics for a single client"""
    accuracy: Optional[float] = None
    loss: Optional[float] = None
    rmse: Optional[float] = None
    mae: Optional[float] = None
    r2: Optional[float] = None

class HistoryEntry(BaseModel):
    """Single entry in the training history"""
    round: int
    global_accuracy: float
    client_metrics: Dict[str, Dict[str, float]]
    completed_at: str
    model_weights: Optional[Dict[str, float]] = None  # GA-Stacking model weights

class TrainingHistory(BaseModel):
    """Training history response"""
    status: str
    history: List[HistoryEntry] = []