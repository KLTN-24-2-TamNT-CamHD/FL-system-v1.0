"""
Enhanced Federated Learning Client with GA-Stacking Ensemble optimization.
Extends the base client with local ensemble optimization capabilities.
"""

import os
import json
import pickle
import time
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from pathlib import Path
from datetime import datetime, timezone
from sklearn.model_selection import train_test_split
import hashlib


import pandas as pd
import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset, random_split
from flwr.common import (
    Parameters,
    Scalar,
    FitIns,
    FitRes,
    EvaluateIns,
    EvaluateRes,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
import numpy as np

from ipfs_connector import IPFSConnector
from blockchain_connector import BlockchainConnector
from ga_stacking import GAStacking, EnsembleModel
from base_models import create_model_ensemble, get_ensemble_state_dict, load_ensemble_from_state_dict, create_model_ensemble_from_config, SklearnModelWrapper, MetaLearnerWrapper

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("FL-Client-Ensemble")


class GAStackingClient(fl.client.NumPyClient):
    """Federated Learning client with GA-Stacking ensemble optimization."""
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        train_loader: DataLoader,
        test_loader: DataLoader,
        ensemble_size: int = 5,
        ipfs_connector: Optional[IPFSConnector] = None,
        blockchain_connector: Optional[BlockchainConnector] = None,
        wallet_address: Optional[str] = None,
        private_key: Optional[str] = None,
        device: str = "cpu",
        client_id: str = None,
        ga_generations: int = 20,
        ga_population_size: int = 30
    ):
        """
        Initialize GA-Stacking client.
        
        Args:
            input_dim: Input dimension for models
            output_dim: Output dimension for models
            train_loader: Training data loader
            test_loader: Test data loader
            ensemble_size: Number of models in the ensemble
            ipfs_connector: IPFS connector
            blockchain_connector: Blockchain connector
            wallet_address: Client's wallet address
            private_key: Client's private key
            device: Device to use for computation
            client_id: Client identifier
            ga_generations: Number of GA generations to run
            ga_population_size: Size of GA population
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.ensemble_size = ensemble_size
        self.ipfs = ipfs_connector or IPFSConnector()
        self.blockchain = blockchain_connector
        self.wallet_address = wallet_address
        self.private_key = private_key
        self.device = torch.device(device)
        self.client_id = client_id or f"client-{os.getpid()}"
        self.ga_generations = ga_generations
        self.ga_population_size = ga_population_size
        
        # Initialize ensemble models
        self.base_models, self.model_names = create_model_ensemble(
            input_dim=input_dim,
            output_dim=output_dim,
            ensemble_size=ensemble_size,
            device=device
        )
        
        # Initialize GA-Stacking optimizer
        self.ga_stacking = None
        self.ensemble_model = None
        
        # Metrics storage
        self.metrics_history = []
        
        logger.info(f"Initialized {self.client_id} with {ensemble_size} base models")
        logger.info(f"IPFS node: {self.ipfs.ipfs_api_url}")
        
        # Verify blockchain authentication if available
        if self.blockchain and self.wallet_address:
            try:
                is_authorized = self.blockchain.is_client_authorized(self.wallet_address)
                if is_authorized:
                    logger.info(f"Client {self.wallet_address} is authorized on the blockchain ✅")
                else:
                    logger.warning(f"Client {self.wallet_address} is NOT authorized on the blockchain ❌")
                    logger.warning("The server may reject this client's contributions")
            except Exception as e:
                logger.error(f"Failed to verify client authorization: {e}")
        
        self.log_system_status()


    def save_ensemble_to_ipfs(self, round_num: int, metrics: Dict[str, Any]) -> str:
        """Save the ensemble model to IPFS with security hash."""
        # Prepare ensemble state
        if self.ensemble_model is None:
            logger.error("No ensemble model to save")
            return None
        
        ensemble_state = self.ensemble_model.get_state_dict()
        
        
        # Create metadata
        model_metadata = {
            "ensemble_state": ensemble_state,
            "info": {
                "round": round_num,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "client_address": self.wallet_address if self.wallet_address else "unknown",
                "accuracy": metrics.get("accuracy", 0.0),
                "loss": metrics.get("loss", 0.0)
            }
        }
        
        # Store in IPFS
        ipfs_hash = self.ipfs.add_json(model_metadata)
        logger.info(f"Stored client ensemble in IPFS: {ipfs_hash}")
        
        # Store security hash in metrics so it can be sent back to server
        metrics["security_hash"] = security_hash
        
        return ipfs_hash
            
def start_client(
    server_address: str = "127.0.0.1:8080",
    ipfs_url: str = "http://127.0.0.1:5001",
    ganache_url: str = "http://127.0.0.1:7545",
    contract_address: Optional[str] = None,
    wallet_address: Optional[str] = None,
    private_key: Optional[str] = None,
    client_id: Optional[str] = None,
    input_dim: int = 10,
    output_dim: int = 1,
    ensemble_size: int = 5,
    device: str = "cpu",
    ga_generations: int = 20,
    ga_population_size: int = 30
) -> None:
    """
    Start a federated learning client with GA-Stacking ensemble optimization.
    
    Args:
        server_address: Server address (host:port)
        ipfs_url: IPFS API URL
        ganache_url: Ganache blockchain URL
        contract_address: Address of deployed EnhancedModelRegistry contract
        wallet_address: Client's Ethereum wallet address
        private_key: Client's private key (for signing transactions)
        client_id: Client identifier
        input_dim: Input dimension for the model
        output_dim: Output dimension for the model
        ensemble_size: Number of models in the ensemble
        device: Device to use for training ('cpu' or 'cuda')
        ga_generations: Number of GA generations to run
        ga_population_size: Size of GA population
    """
    # Create client ID if not provided
    if client_id is None:
        client_id = f"client-{os.getpid()}"
    
    # Create metrics directory
    os.makedirs(f"metrics/{client_id}", exist_ok=True)
    
    # Initialize IPFS connector
    ipfs_connector = IPFSConnector(ipfs_api_url=ipfs_url)
    logger.info(f"Initialized IPFS connector: {ipfs_url}")
    
    # Initialize blockchain connector if contract address is provided
    blockchain_connector = None
    if contract_address:
        try:
            blockchain_connector = BlockchainConnector(
                ganache_url=ganache_url,
                contract_address=contract_address,
                private_key=private_key
            )
            logger.info(f"Initialized blockchain connector: {ganache_url}")
            logger.info(f"Using contract at: {contract_address}")
        except Exception as e:
            logger.error(f"Failed to initialize blockchain connector: {e}")
            logger.warning("Continuing without blockchain features")
    
    # Load dataset from files
    def load_dataset_from_files(train_file, test_file):
        # Read train and test files
        train_df = pd.read_csv(train_file, comment='#')
        test_df = pd.read_csv(test_file, comment='#')
        
        # Extract features and targets
        feature_columns = [col for col in train_df.columns if col != 'target']
        
        # Convert to tensors
        train_x = torch.tensor(train_df[feature_columns].values, dtype=torch.float32)
        train_y = torch.tensor(train_df['target'].values.reshape(-1, 1), dtype=torch.float32)
        
        test_x = torch.tensor(test_df[feature_columns].values, dtype=torch.float32)
        test_y = torch.tensor(test_df['target'].values.reshape(-1, 1), dtype=torch.float32)
        
        # Update input and output dimensions based on the data
        nonlocal input_dim, output_dim
        input_dim = train_x.shape[1]
        output_dim = train_y.shape[1]
        
        logger.info(f"Dataset loaded - Input dim: {input_dim}, Output dim: {output_dim}")
        
        # Create data loaders
        class SimpleDataset(torch.utils.data.Dataset):
            def __init__(self, x, y):
                self.x = x
                self.y = y
            
            def __len__(self):
                return len(self.x)
            
            def __getitem__(self, idx):
                return self.x[idx], self.y[idx]
        
        train_loader = DataLoader(
            SimpleDataset(train_x, train_y), batch_size=32, shuffle=True
        )
        test_loader = DataLoader(
            SimpleDataset(test_x, test_y), batch_size=32, shuffle=False
        )
        
        return train_loader, test_loader

    # Determine which dataset files to load based on client_id
    train_file = f"{client_id}_train.txt"
    test_file = f"{client_id}_test.txt"

    # Check if files exist
    if not os.path.exists(train_file) or not os.path.exists(test_file):
        logger.warning(f"Dataset files not found for {client_id}, using default files")
        train_file = "client-1_train.txt"  # Fallback to client-1 if files not found
        test_file = "client-1_test.txt"

    # Load dataset
    train_loader, test_loader = load_dataset_from_files(train_file, test_file)
    
    # Create client
    client = GAStackingClient(
        input_dim=input_dim,
        output_dim=output_dim,
        train_loader=train_loader,
        test_loader=test_loader,
        ensemble_size=ensemble_size,
        ipfs_connector=ipfs_connector,
        blockchain_connector=blockchain_connector,
        wallet_address=wallet_address,
        private_key=private_key,
        device=device,
        client_id=client_id,
        ga_generations=ga_generations,
        ga_population_size=ga_population_size
    )
    
    # Start client
    fl.client.start_client(server_address=server_address, client=client)
    
    # Save metrics after client finishes
    client.save_metrics_history(filepath=f"metrics/{client_id}/metrics_history.json")
    
    logger.info(f"Client {client_id} completed federated learning with GA-Stacking")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Start FL client with GA-Stacking ensemble optimization")
    parser.add_argument("--server-address", type=str, default="127.0.0.1:8088", help="Server address (host:port)")
    parser.add_argument("--ipfs-url", type=str, default="http://127.0.0.1:5001/api/v0", help="IPFS API URL")
    parser.add_argument("--ganache-url", type=str, default="http://192.168.1.146:7545", help="Ganache blockchain URL")
    parser.add_argument("--contract-address", type=str, help="Address of deployed EnhancedModelRegistry contract")
    parser.add_argument("--wallet-address", type=str, help="Client's Ethereum wallet address")
    parser.add_argument("--private-key", type=str, help="Client's private key (for signing transactions)")
    parser.add_argument("--client-id", type=str, help="Client identifier")
    parser.add_argument("--input-dim", type=int, default=10, help="Input dimension for the model")
    parser.add_argument("--output-dim", type=int, default=1, help="Output dimension for the model")
    parser.add_argument("--ensemble-size", type=int, default=5, help="Number of models in the ensemble")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device for training")
    parser.add_argument("--ga-generations", type=int, default=20, help="Number of GA generations to run")
    parser.add_argument("--ga-population-size", type=int, default=30, help="Size of GA population")
    parser.add_argument("--train-file", type=str, help="Path to training data file")
    parser.add_argument("--test-file", type=str, help="Path to test data file")
    
    args = parser.parse_args()
    
    # Check if contract address is stored in file
    if args.contract_address is None:
        try:
            with open("contract_address.txt", "r") as f:
                args.contract_address = f.read().strip()
                print(f"Loaded contract address from file: {args.contract_address}")
        except FileNotFoundError:
            print("No contract address provided or found in file")
    
    # Override train/test files if provided
    if args.train_file and args.test_file:
        train_file = args.train_file
        test_file = args.test_file
    start_client(
        server_address=args.server_address,
        ipfs_url=args.ipfs_url,
        ganache_url=args.ganache_url,
        contract_address=args.contract_address,
        wallet_address=args.wallet_address,
        private_key=args.private_key,
        client_id=args.client_id,
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        ensemble_size=args.ensemble_size,
        device=args.device,
        ga_generations=args.ga_generations,
        ga_population_size=args.ga_population_size
    )