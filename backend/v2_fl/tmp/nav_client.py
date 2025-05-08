# """
# Enhanced Federated Learning Client with IPFS integration and blockchain authentication.
# Supports wallet-based authentication and tracking of client contributions.
# """

# import os
# import json
# import pickle
# import time
# from typing import Dict, List, Optional, Tuple, Union, Any
# import logging
# from pathlib import Path
# from datetime import datetime, timezone

# import flwr as fl
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from flwr.common import (
#     Parameters,
#     Scalar,
#     FitIns,
#     FitRes,
#     EvaluateIns,
#     EvaluateRes,
#     ndarrays_to_parameters,
#     parameters_to_ndarrays,
# )
# import numpy as np

# from ipfs_connector import IPFSConnector
# from blockchain_connector import BlockchainConnector

# # Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
# )
# logger = logging.getLogger("FL-Client")


# class SimpleLinearModel(nn.Module):
#     """Simple linear model for demonstration purposes."""
    
#     def __init__(self, input_dim=10, output_dim=1):
#         super(SimpleLinearModel, self).__init__()
#         self.linear = nn.Linear(input_dim, output_dim)
        
#     def forward(self, x):
#         return self.linear(x)


# class EnhancedClient(fl.client.NumPyClient):
#     """Federated Learning client with IPFS integration and blockchain authentication."""
    
#     def __init__(
#         self,
#         model: nn.Module,
#         train_loader: DataLoader,
#         test_loader: DataLoader,
#         ipfs_connector: Optional[IPFSConnector] = None,
#         blockchain_connector: Optional[BlockchainConnector] = None,
#         wallet_address: Optional[str] = None,
#         private_key: Optional[str] = None,
#         device: str = "cpu",
#         client_id: str = None,
#     ):
#         self.model = model
#         self.train_loader = train_loader
#         self.test_loader = test_loader
#         self.ipfs = ipfs_connector or IPFSConnector()
#         self.blockchain = blockchain_connector
#         self.wallet_address = wallet_address
#         self.private_key = private_key
#         self.device = torch.device(device)
#         self.model.to(self.device)
#         self.client_id = client_id or f"client-{os.getpid()}"
        
#         # Metrics storage
#         self.metrics_history = []
        
#         logger.info(f"Initialized {self.client_id} with IPFS node: {self.ipfs.ipfs_api_url}")
        
#         # Verify blockchain authentication if available
#         if self.blockchain and self.wallet_address:
#             try:
#                 is_authorized = self.blockchain.is_client_authorized(self.wallet_address)
#                 if is_authorized:
#                     logger.info(f"Client {self.wallet_address} is authorized on the blockchain ✅")
#                 else:
#                     logger.warning(f"Client {self.wallet_address} is NOT authorized on the blockchain ❌")
#                     logger.warning("The server may reject this client's contributions")
#             except Exception as e:
#                 logger.error(f"Failed to verify client authorization: {e}")
    
#     def get_parameters(self, config: Dict[str, Scalar]) -> List[np.ndarray]:
#         """Get model parameters as a list of NumPy arrays."""
#         return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
#     def set_parameters(self, parameters: List[np.ndarray]) -> None:
#         """Set model parameters from a list of NumPy arrays."""
#         params_dict = zip(self.model.state_dict().keys(), parameters)
#         state_dict = {k: torch.Tensor(v) for k, v in params_dict}
#         self.model.load_state_dict(state_dict, strict=True)
    
#     def set_parameters_from_ipfs(self, ipfs_hash: str) -> None:
#         """Set model parameters from IPFS."""
#         try:
#             # Get model data from IPFS
#             model_data = self.ipfs.get_json(ipfs_hash)
            
#             if model_data and "state_dict" in model_data:
#                 # Load state dict
#                 state_dict = {
#                     k: torch.tensor(v, device=self.device)
#                     for k, v in model_data["state_dict"].items()
#                 }
                
#                 # Update model
#                 self.model.load_state_dict(state_dict)
#                 logger.info(f"Model loaded from IPFS: {ipfs_hash}")
                
#                 # Log metadata
#                 if "info" in model_data:
#                     logger.info(f"Model info: {model_data['info']}")
#             else:
#                 logger.error(f"Invalid model data from IPFS: {ipfs_hash}")
                
#         except Exception as e:
#             logger.error(f"Failed to load model from IPFS: {e}")
    
#     def save_parameters_to_ipfs(
#         self, parameters: List[np.ndarray], round_num: int, metrics: Dict[str, Scalar] = None
#     ) -> str:
#         """Save model parameters to IPFS."""
#         # Create state dict from parameters
#         state_dict = {}
#         for i, key in enumerate(self.model.state_dict().keys()):
#             if i < len(parameters):
#                 state_dict[key] = parameters[i].tolist()
        
#         # Create metadata
#         model_metadata = {
#             "state_dict": state_dict,
#             "info": {
#                 "round": round_num,
#                 "client_id": self.client_id,
#                 "wallet_address": self.wallet_address if self.wallet_address else "unknown",
#                 "timestamp": datetime.now(timezone.utc).isoformat()
#             }
#         }
        
#         # Add metrics if provided
#         if metrics:
#             model_metadata["info"]["metrics"] = metrics
        
#         # Store in IPFS
#         ipfs_hash = self.ipfs.add_json(model_metadata)
#         logger.info(f"Stored model in IPFS: {ipfs_hash}")
        
#         return ipfs_hash
    
#     def fit(self, parameters: Parameters, config: Dict[str, Scalar]) -> Tuple[Parameters, int, Dict[str, Scalar]]:
#         """Train the model on the local dataset."""
#         # Check if client is authorized (if blockchain is available)
#         if self.blockchain and self.wallet_address:
#             try:
#                 is_authorized = self.blockchain.is_client_authorized(self.wallet_address)
#                 if not is_authorized:
#                     logger.warning(f"Client {self.wallet_address} is not authorized to participate")
#                     # Return empty parameters with auth failure flag
#                     return parameters, 0, {"error": "client_not_authorized"}
#             except Exception as e:
#                 logger.error(f"Failed to check client authorization: {e}")
#                 # Continue anyway, server will check again
        
#         # Get global model from IPFS if hash is provided
#         ipfs_hash = config.get("ipfs_hash", None)
#         round_num = config.get("server_round", 0)
        
#         if ipfs_hash:
#             self.set_parameters_from_ipfs(ipfs_hash)
#         else:
#             # Fallback to direct parameters
#             self.set_parameters(parameters_to_ndarrays(parameters))
        
#         # Get training config
#         epochs = int(config.get("epochs", 1))
#         learning_rate = float(config.get("learning_rate", 0.01))
        
#         # Define loss function and optimizer
#         criterion = nn.MSELoss()
#         optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)
        
#         # Training loop
#         self.model.train()
#         train_loss = 0.0
#         num_samples = 0
        
#         for epoch in range(epochs):
#             epoch_loss = 0.0
#             epoch_samples = 0
            
#             for batch_idx, (data, target) in enumerate(self.train_loader):
#                 data, target = data.to(self.device), target.to(self.device)
                
#                 optimizer.zero_grad()
#                 output = self.model(data)
#                 loss = criterion(output, target)
#                 loss.backward()
#                 optimizer.step()
                
#                 epoch_loss += loss.item() * len(data)
#                 epoch_samples += len(data)
            
#             avg_epoch_loss = epoch_loss / epoch_samples
#             logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_epoch_loss:.4f}")
            
#             train_loss += epoch_loss
#             num_samples += epoch_samples
        
#         # Calculate average loss
#         avg_loss = train_loss / num_samples if num_samples > 0 else None
        
#         # Get updated parameters
#         parameters_updated = self.get_parameters(config={})
        
#         # Save model to IPFS
#         accuracy = self._calculate_accuracy()
#         metrics = {
#             "loss": float(avg_loss) if avg_loss is not None else 0.0,
#             "accuracy": float(accuracy)
#         }
#         client_ipfs_hash = self.save_parameters_to_ipfs(parameters_updated, round_num, metrics)
        
#         # Record contribution to blockchain if available
#         if self.blockchain and self.wallet_address:
#             try:
#                 tx_hash = self.blockchain.record_contribution(
#                     client_address=self.wallet_address,
#                     round_num=round_num,
#                     ipfs_hash=client_ipfs_hash,
#                     accuracy=accuracy
#                 )
#                 logger.info(f"Contribution recorded on blockchain, tx: {tx_hash}")
#                 metrics["blockchain_tx"] = tx_hash
#             except Exception as e:
#                 logger.error(f"Failed to record contribution on blockchain: {e}")
        
#         # Add to metrics history
#         self.metrics_history.append({
#             "round": round_num,
#             "fit_loss": float(avg_loss) if avg_loss is not None else 0.0,
#             "accuracy": float(accuracy),
#             "train_samples": num_samples,
#             "ipfs_hash": client_ipfs_hash,
#             "wallet_address": self.wallet_address if self.wallet_address else "unknown",
#             "timestamp": datetime.now(timezone.utc).isoformat()
#         })
        
#         # Include IPFS hash in metrics
#         metrics["ipfs_hash"] = ipfs_hash  # Return the server's hash for tracking
#         metrics["client_ipfs_hash"] = client_ipfs_hash
#         metrics["wallet_address"] = self.wallet_address if self.wallet_address else "unknown"
        
#         # Convert parameters to Flower format
#         parameters_proto = ndarrays_to_parameters(parameters_updated)

#         # Make sure metrics only contains serializable values
#         for key in list(metrics.keys()):
#             if not isinstance(metrics[key], (int, float, str, bool)):
#                 metrics[key] = str(metrics[key])

#         # Return the exact expected format
#         return parameters_updated, num_samples, metrics
    
#     def evaluate(
#         self, parameters: Parameters, config: Dict[str, Scalar]
#     ) -> Tuple[float, int, Dict[str, Scalar]]:
#         """Evaluate the model on the local test dataset."""
#         # Check if client is authorized (if blockchain is available)
#         if self.blockchain and self.wallet_address:
#             try:
#                 is_authorized = self.blockchain.is_client_authorized(self.wallet_address)
#                 if not is_authorized:
#                     logger.warning(f"Client {self.wallet_address} is not authorized to participate in evaluation")
#                     # Return empty evaluation with auth failure flag
#                     return 0.0, 0, {"error": "client_not_authorized"}
#             except Exception as e:
#                 logger.error(f"Failed to check client authorization: {e}")
#                 # Continue anyway, server will check again
        
#         # Get global model from IPFS if hash is provided
#         ipfs_hash = config.get("ipfs_hash", None)
#         round_num = config.get("server_round", 0)
        
#         if ipfs_hash:
#             self.set_parameters_from_ipfs(ipfs_hash)
#         else:
#             # Fallback to direct parameters
#             self.set_parameters(parameters_to_ndarrays(parameters))
        
#         # Evaluation loop
#         self.model.eval()
#         test_loss = 0.0
#         num_samples = 0
#         criterion = nn.MSELoss()
        
#         with torch.no_grad():
#             for data, target in self.test_loader:
#                 data, target = data.to(self.device), target.to(self.device)
                
#                 output = self.model(data)
#                 loss = criterion(output, target)
                
#                 test_loss += loss.item() * len(data)
#                 num_samples += len(data)
        
#         # Calculate average loss
#         avg_loss = test_loss / num_samples if num_samples > 0 else None
        
#         # Calculate accuracy
#         accuracy = self._calculate_accuracy()
        
#         # Add to metrics history
#         self.metrics_history.append({
#             "round": round_num,
#             "eval_loss": float(avg_loss) if avg_loss is not None else 0.0,
#             "accuracy": float(accuracy),
#             "eval_samples": num_samples,
#             "ipfs_hash": ipfs_hash,
#             "wallet_address": self.wallet_address if self.wallet_address else "unknown",
#             "timestamp": datetime.now(timezone.utc).isoformat()
#         })
        
#         metrics = {
#             "loss": float(avg_loss) if avg_loss is not None else 0.0,
#             "accuracy": float(accuracy),
#             "wallet_address": self.wallet_address if self.wallet_address else "unknown"
#         }
        
#         return float(avg_loss) if avg_loss is not None else 0.0, num_samples, metrics
    
#     def _calculate_accuracy(self) -> float:
#         """Calculate accuracy on the test dataset."""
#         self.model.eval()
#         correct = 0
#         total = 0
        
#         with torch.no_grad():
#             for data, target in self.test_loader:
#                 data, target = data.to(self.device), target.to(self.device)
                
#                 outputs = self.model(data)
                
#                 # For regression, we consider a prediction correct if it's within a threshold
#                 threshold = 0.5
#                 correct += torch.sum((torch.abs(outputs - target) < threshold)).item()
#                 total += target.size(0)
        
#         accuracy = 100.0 * correct / total if total > 0 else 0.0
#         logger.info(f"Accuracy: {accuracy:.2f}%")
        
#         return accuracy
    
#     def save_metrics_history(self, filepath: str = "client_metrics.json"):
#         """Save metrics history to a file."""
#         with open(filepath, "w") as f:
#             json.dump(self.metrics_history, f, indent=2)
#         logger.info(f"Saved metrics history to {filepath}")


# def start_client(
#     server_address: str = "127.0.0.1:8080",
#     ipfs_url: str = "http://127.0.0.1:5001",
#     ganache_url: str = "http://127.0.0.1:7545",
#     contract_address: Optional[str] = None,
#     wallet_address: Optional[str] = None,
#     private_key: Optional[str] = None,
#     client_id: Optional[str] = None,
#     input_dim: int = 10,
#     output_dim: int = 1,
#     device: str = "cpu"
# ) -> None:
#     """
#     Start a federated learning client with blockchain authentication.
    
#     Args:
#         server_address: Server address (host:port)
#         ipfs_url: IPFS API URL
#         ganache_url: Ganache blockchain URL
#         contract_address: Address of deployed EnhancedModelRegistry contract
#         wallet_address: Client's Ethereum wallet address
#         private_key: Client's private key (for signing transactions)
#         client_id: Client identifier
#         input_dim: Input dimension for the model
#         output_dim: Output dimension for the model
#         device: Device to use for training ('cpu' or 'cuda')
#     """
#     # Create client ID if not provided
#     if client_id is None:
#         client_id = f"client-{os.getpid()}"
    
#     # Create metrics directory
#     os.makedirs(f"metrics/{client_id}", exist_ok=True)
    
#     # Initialize IPFS connector
#     ipfs_connector = IPFSConnector(ipfs_api_url=ipfs_url)
#     logger.info(f"Initialized IPFS connector: {ipfs_url}")
    
#     # Initialize blockchain connector if contract address is provided
#     blockchain_connector = None
#     if contract_address:
#         try:
#             blockchain_connector = BlockchainConnector(
#                 ganache_url=ganache_url,
#                 contract_address=contract_address,
#                 private_key=private_key
#             )
#             logger.info(f"Initialized blockchain connector: {ganache_url}")
#             logger.info(f"Using contract at: {contract_address}")
#         except Exception as e:
#             logger.error(f"Failed to initialize blockchain connector: {e}")
#             logger.warning("Continuing without blockchain features")
    
#     # Create model
#     model = SimpleLinearModel(input_dim=input_dim, output_dim=output_dim)
    
#     # Create synthetic data for demonstration
#     # In a real application, you would load your actual dataset
#     def generate_synthetic_data(num_samples=100):
#         x = torch.randn(num_samples, input_dim)
#         w = torch.randn(input_dim, output_dim)
#         b = torch.randn(output_dim)
#         y = torch.matmul(x, w) + b + 0.1 * torch.randn(num_samples, output_dim)
#         return x, y
    
#     # Generate data
#     train_x, train_y = generate_synthetic_data(100)
#     test_x, test_y = generate_synthetic_data(20)
    
#     # Create data loaders
#     class SimpleDataset(torch.utils.data.Dataset):
#         def __init__(self, x, y):
#             self.x = x
#             self.y = y
        
#         def __len__(self):
#             return len(self.x)
        
#         def __getitem__(self, idx):
#             return self.x[idx], self.y[idx]
    
#     train_loader = DataLoader(
#         SimpleDataset(train_x, train_y), batch_size=10, shuffle=True
#     )
#     test_loader = DataLoader(
#         SimpleDataset(test_x, test_y), batch_size=10, shuffle=False
#     )
    
#     # Create client
#     client = EnhancedClient(
#         model=model,
#         train_loader=train_loader,
#         test_loader=test_loader,
#         ipfs_connector=ipfs_connector,
#         blockchain_connector=blockchain_connector,
#         wallet_address=wallet_address,
#         private_key=private_key,
#         device=device,
#         client_id=client_id
#     )
    
#     # Start client
#     fl.client.start_client(server_address=server_address, client=client)
    
#     # Save metrics after client finishes
#     client.save_metrics_history(filepath=f"metrics/{client_id}/metrics_history.json")
    
#     logger.info(f"Client {client_id} completed federated learning")


# if __name__ == "__main__":
#     import argparse
    
#     parser = argparse.ArgumentParser(description="Start FL client with IPFS and blockchain integration")
#     parser.add_argument("--server-address", type=str, default="127.0.0.1:8080", help="Server address (host:port)")
#     parser.add_argument("--ipfs-url", type=str, default="http://127.0.0.1:5001", help="IPFS API URL")
#     parser.add_argument("--ganache-url", type=str, default="http://127.0.0.1:7545", help="Ganache blockchain URL")
#     parser.add_argument("--contract-address", type=str, help="Address of deployed EnhancedModelRegistry contract")
#     parser.add_argument("--wallet-address", type=str, help="Client's Ethereum wallet address")
#     parser.add_argument("--private-key", type=str, help="Client's private key (for signing transactions)")
#     parser.add_argument("--client-id", type=str, help="Client identifier")
#     parser.add_argument("--input-dim", type=int, default=10, help="Input dimension for the model")
#     parser.add_argument("--output-dim", type=int, default=1, help="Output dimension for the model")
#     parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device for training")
    
#     args = parser.parse_args()
    
#     # Check if contract address is stored in file
#     if args.contract_address is None:
#         try:
#             with open("contract_address.txt", "r") as f:
#                 args.contract_address = f.read().strip()
#                 print(f"Loaded contract address from file: {args.contract_address}")
#         except FileNotFoundError:
#             print("No contract address provided or found in file")
    
#     start_client(
#         server_address=args.server_address,
#         ipfs_url=args.ipfs_url,
#         ganache_url=args.ganache_url,
#         contract_address=args.contract_address,
#         wallet_address=args.wallet_address,
#         private_key=args.private_key,
#         client_id=args.client_id,
#         input_dim=args.input_dim,
#         output_dim=args.output_dim,
#         device=args.device
#     )
