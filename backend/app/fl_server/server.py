#!/usr/bin/env python3
import argparse
import json
import os
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

import flwr as fl
import numpy as np
import torch
from flwr.common import (
    EvaluateRes,
    FitRes,
    Parameters,
    Scalar,
    parameters_to_ndarrays,
    ndarrays_to_parameters,
)
from flwr.server.client_proxy import ClientProxy

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.blockchain_service import BlockchainService
from services.ipfs_service import IPFSService
from utils.server_logger import ServerLogger

# Constants
DEFAULT_SERVER_ADDRESS = "0.0.0.0:8088"
DEFAULT_CONFIG_FILE = "config.json"
DEFAULT_LOG_DIR = "logs"

class FraudDetectionServer:
    """Federated fraud detection server with blockchain and IPFS integration."""
    
    def __init__(
        self,
        server_address: str,
        config: Dict[str, Any],
        log_dir: str,
        model_path: Optional[str] = None
    ):
        """
        Initialize the federated server.
        
        Args:
            server_address: Server address (host:port)
            config: Configuration dictionary
            log_dir: Directory for logs
            model_path: Path to initial model (if any)
        """
        self.server_address = server_address
        self.config = config
        self.log_dir = log_dir
        self.model_path = model_path
        
        # Set up logger
        self.logger = ServerLogger(
            log_dir=log_dir,
            experiment_name=config.get("experiment_name", f"fraud_detection_{int.from_bytes(os.urandom(4), 'big')}")
        )
        
        # Initialize blockchain service
        if "blockchain" in config:
            self.bc_service = BlockchainService(
                provider_url=config["blockchain"]["provider_url"],
                contract_address=config["blockchain"]["contract_address"],
                contract_abi_path=config["blockchain"]["contract_abi_path"],
                private_key=config["blockchain"].get("private_key"),
                account_address=config["blockchain"].get("account_address")
            )
        else:
            self.bc_service = None
            self.logger.logger.warning("Blockchain service not configured")
        
        # Initialize IPFS service
        if "ipfs" in config:
            self.ipfs_service = IPFSService(
                api_url=config["ipfs"]["api_url"]
            )
        else:
            self.ipfs_service = None
            self.logger.logger.warning("IPFS service not configured")
            
        # Current round tracking
        self.current_round = 0
        self.global_model_hash = None
        
        # Initialize Flower strategies
        self.strategy = self._create_strategy()
        
        # Initialize server app configuration
        server_config = fl.server.ServerConfig(num_rounds=config.get("num_rounds", 3))
        
        # Initialize parent class - removed since we're not extending ServerApp anymore
        # super().__init__(
        #    server_address=server_address,
        #    config=server_config,
        #    strategy=self.strategy
        # )
        
        # Log initialization
        self.logger.logger.info(f"Server initialized at {server_address}")
        self.logger.logger.info(f"Configuration: {json.dumps(config, default=str)}")
    
    def _create_strategy(self) -> fl.server.strategy.Strategy:
        """Create the federated learning strategy."""
        
        def fit_config_fn(server_round: int) -> Dict[str, Any]:
            """Return training configuration for clients."""
            config = {
                "round": server_round,
                "epochs": self.config.get("client_epochs", 1),
                "batch_size": self.config.get("batch_size", 32),
                "learning_rate": self.config.get("learning_rate", 0.001),
                "global_model_hash": self.global_model_hash
            }
            return config
        
        def evaluate_config_fn(server_round: int) -> Dict[str, Any]:
            """Return evaluation configuration for clients."""
            return {
                "round": server_round,
                "batch_size": self.config.get("batch_size", 32),
                "global_model_hash": self.global_model_hash
            }
        
        # Define custom functions for handling client properties and metrics
        def on_fit_config_fn(server_round: int) -> Dict[str, Any]:
            """Called before sending fit configurations to clients."""
            # Start a new training round on the blockchain
            if self.bc_service:
                try:
                    tx_hash, round_id = self.bc_service.initiate_training_round()
                    self.logger.log_round_start(round_id, tx_hash)
                    self.current_round = round_id
                except Exception as e:
                    self.logger.logger.error(f"Failed to initiate training round: {e}")
                    self.current_round = server_round
                    self.logger.log_round_start(server_round)
            else:
                self.current_round = server_round
                self.logger.log_round_start(server_round)
            
            return fit_config_fn(server_round)
        
        def fit_metrics_aggregation_fn(metrics: List[Tuple[int, Dict[str, Scalar]]]) -> Dict[str, Scalar]:
            """Aggregate fit metrics from clients."""
            # Here you can process and aggregate metrics as needed
            aggregated = {}
            
            if not metrics:
                return aggregated
                
            # Calculate sum and count for each metric
            for _, m in metrics:
                for k, v in m.items():
                    if k not in aggregated:
                        aggregated[k] = {"sum": 0.0, "count": 0}
                    aggregated[k]["sum"] += v
                    aggregated[k]["count"] += 1
            
            # Calculate average for each metric
            return {k: v["sum"] / v["count"] for k, v in aggregated.items()}
        
        def evaluate_metrics_aggregation_fn(metrics: List[Tuple[int, Dict[str, Scalar]]]) -> Dict[str, Scalar]:
            """Aggregate evaluation metrics from clients."""
            # Similar to fit metrics aggregation
            return fit_metrics_aggregation_fn(metrics)
        
        # Parameters initialization function
        parameters = None
        # Hardcode initial parameters instead of loading from file
        try:
            # Create a simple FraudDetectionModel structure with hardcoded parameters
            input_dim = self.config.get("model", {}).get("input_dim", 50)
            hidden_dims = self.config.get("model", {}).get("hidden_dims", [32, 16])
            output_dim = self.config.get("model", {}).get("output_dim", 1)
            
            self.logger.logger.info(f"Initializing model with input_dim={input_dim}, hidden_dims={hidden_dims}, output_dim={output_dim}")
            
            # Create initial parameters for each layer
            initial_params = []
            
            # First layer: input_dim -> hidden_dims[0]
            weight = np.random.randn(hidden_dims[0], input_dim).astype(np.float32) * 0.1
            bias = np.zeros(hidden_dims[0]).astype(np.float32)
            initial_params.append(weight)
            initial_params.append(bias)
            
            # Hidden layers
            for i in range(len(hidden_dims) - 1):
                # Linear layer
                weight = np.random.randn(hidden_dims[i+1], hidden_dims[i]).astype(np.float32) * 0.1
                bias = np.zeros(hidden_dims[i+1]).astype(np.float32)
                initial_params.append(weight)
                initial_params.append(bias)
            
            # Output layer: hidden_dims[-1] -> output_dim
            weight = np.random.randn(output_dim, hidden_dims[-1]).astype(np.float32) * 0.1
            bias = np.zeros(output_dim).astype(np.float32)
            initial_params.append(weight)
            initial_params.append(bias)
            
            # Convert to Flower parameters
            parameters = ndarrays_to_parameters(initial_params)
            
            self.logger.logger.info(f"Successfully created initial model parameters with {len(initial_params)} tensors")
            
            # Store initial model to IPFS if available
            if self.ipfs_service:
                self.global_model_hash = self.ipfs_service.store_model_params(
                    initial_params,
                    metadata={"round": 0, "type": "initial"}
                )
                self.logger.logger.info(f"Initial model stored to IPFS: {self.global_model_hash}")
                
        except Exception as e:
            self.logger.logger.error(f"Failed to create initial model parameters: {e}")
            self.logger.logger.info("Will use random parameters from a client instead")
        
        # We'll implement callbacks later if needed
        self.callback = None
        
        # Configure the strategy
        strategy = fl.server.strategy.FedAvg(
            min_fit_clients=self.config.get("min_fit_clients", 2),
            min_available_clients=self.config.get("min_available_clients", 2),
            min_evaluate_clients=self.config.get("min_evaluate_clients", 2),
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=evaluate_config_fn,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
            initial_parameters=parameters,
            accept_failures=self.config.get("accept_failures", True)
        )
        
        # Extend the strategy with custom functionality
        original_aggregate_fit = strategy.aggregate_fit
        
        def extended_aggregate_fit(
            server_round: int,
            results: List[Tuple[ClientProxy, FitRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]]
        ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
            """Extended aggregation function that integrates with blockchain and IPFS."""
            # First, perform the original aggregation
            parameters, metrics = original_aggregate_fit(server_round, results, failures)
            
            if parameters is not None:
                # Convert parameters to numpy arrays
                model_params = parameters_to_ndarrays(parameters)
                
                # Store the aggregated model on IPFS
                if self.ipfs_service:
                    try:
                        ipfs_hash = self.ipfs_service.store_model_params(
                            model_params,
                            metadata={
                                "round": server_round,
                                "type": "aggregated",
                                "metrics": metrics
                            }
                        )
                        self.global_model_hash = ipfs_hash
                        
                        # Log round aggregation
                        self.logger.log_round_aggregation(
                            round_id=self.current_round,
                            global_model_hash=ipfs_hash,
                            metrics=metrics
                        )
                        
                        # Update blockchain if configured
                        if self.bc_service:
                            try:
                                tx_hash = self.bc_service.completeRound(
                                    self.current_round,
                                    ipfs_hash
                                )
                                
                                # Log round completion
                                self.logger.log_round_completion(
                                    round_id=self.current_round,
                                    global_model_hash=ipfs_hash,
                                    tx_hash=tx_hash,
                                    metrics=metrics
                                )
                            except Exception as e:
                                self.logger.logger.error(f"Failed to complete round on blockchain: {e}")
                    except Exception as e:
                        self.logger.logger.error(f"Failed to store model on IPFS: {e}")
            
            return parameters, metrics
        
        # Replace the aggregate_fit method
        strategy.aggregate_fit = extended_aggregate_fit
        
        # We've removed the CustomCallback for now
        # self.callback = CustomCallback(self)
        self.callback = None
        
        return strategy
    
    # Custom handler for client updates
    def handle_client_update(self, client_proxy: ClientProxy, fit_res: FitRes, client_id: str) -> None:
        """Handle client update with blockchain/IPFS integration."""
        # Extract client properties if available
        client_properties = getattr(client_proxy, "properties", {})
        client_address = client_properties.get("address", "unknown")
        
        # Add to logger
        self.logger.log_client_participation(
            client_id=client_id,
            client_address=client_address,
            round_id=self.current_round,
            properties=client_properties
        )
        
        # Process the model update
        if self.ipfs_service:
            try:
                # Extract model parameters
                model_params = parameters_to_ndarrays(fit_res.parameters)
                
                # Store on IPFS
                ipfs_hash = self.ipfs_service.store_model_params(
                    model_params,
                    metadata={
                        "client_id": client_id,
                        "round": self.current_round,
                        "metrics": fit_res.metrics,
                        "type": "client_update"
                    }
                )
                
                # Submit to blockchain if configured
                tx_hash = None
                if self.bc_service and client_address != "unknown":
                    try:
                        metrics_json = json.dumps(fit_res.metrics, default=str)
                        tx_hash = self.bc_service.submit_model_update(
                            self.current_round,
                            ipfs_hash,
                            metrics_json
                        )
                    except Exception as e:
                        self.logger.logger.error(f"Failed to submit model update to blockchain: {e}")
                
                # Log the update
                self.logger.log_client_update(
                    client_id=client_id,
                    round_id=self.current_round,
                    ipfs_hash=ipfs_hash,
                    tx_hash=tx_hash,
                    metrics=fit_res.metrics
                )
                
            except Exception as e:
                self.logger.logger.error(f"Failed to process client update: {e}")
    
    @classmethod
    def load_config(cls, config_path: str) -> Dict[str, Any]:
        """Load configuration from a JSON file."""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading config: {e}")
            return {}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Federated Fraud Detection Server")
    parser.add_argument(
        "--server-address", 
        type=str, 
        default=DEFAULT_SERVER_ADDRESS,
        help=f"Server address (host:port), default: {DEFAULT_SERVER_ADDRESS}"
    )
    parser.add_argument(
        "--config", 
        type=str, 
        default=DEFAULT_CONFIG_FILE,
        help=f"Path to configuration file, default: {DEFAULT_CONFIG_FILE}"
    )
    parser.add_argument(
        "--log-dir", 
        type=str, 
        default=DEFAULT_LOG_DIR,
        help=f"Directory for logs, default: {DEFAULT_LOG_DIR}"
    )
    parser.add_argument(
        "--model-path", 
        type=str, 
        default=None,
        help="Path to initial model file (optional)"
    )
    return parser.parse_args()


def main():
    """Main function to run the server."""
    args = parse_args()
    
    # Create log directory if it doesn't exist
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Load configuration
    config = FraudDetectionServer.load_config(args.config)
    
    # Create and start the server
    server = FraudDetectionServer(
        server_address=args.server_address,
        config=config,
        log_dir=args.log_dir,
        model_path=args.model_path
    )
    
    try:
        # Start the server
        fl.server.start_server(
            server_address=args.server_address,
            config=fl.server.ServerConfig(num_rounds=config.get("num_rounds", 3)),
            strategy=server.strategy
        )
    except KeyboardInterrupt:
        print("Server stopped by user")
    except Exception as e:
        print(f"Error starting server: {e}")
    finally:
        # Save metrics before exiting
        server.logger.save_metrics()


if __name__ == "__main__":
    main()