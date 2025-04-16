#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
import logging
from typing import Dict, List, Tuple, Union, Optional, Any, Callable
from collections import OrderedDict

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from collections import OrderedDict
from flwr.common import (
    Code,
    Config,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    GetPropertiesIns,
    GetPropertiesRes,
    Status,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.blockchain_service import BlockchainService
from services.ipfs_service import IPFSService

# Constants
DEFAULT_SERVER_ADDRESS = "127.0.0.1:8088"
DEFAULT_CONFIG_FILE = "client_config.json"

class FraudDetectionModel(nn.Module):
    """Neural network model for fraud detection."""
    
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int = 1):
        """
        Initialize the fraud detection model.
        
        Args:
            input_dim: Input dimension
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension (1 for binary classification)
        """
        super(FraudDetectionModel, self).__init__()
        
        # Create layers explicitly to ensure parameter order matches server
        self.layers = nn.ModuleList()
        
        # First layer
        self.layers.append(nn.Linear(input_dim, hidden_dims[0]))
        
        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            self.layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
        
        # Output layer
        self.layers.append(nn.Linear(hidden_dims[-1], output_dim))
        
        # Activation functions (these don't have parameters)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""
        # First layer
        x = self.layers[0](x)
        x = self.relu(x)
        
        # Hidden layers
        for i in range(1, len(self.layers) - 1):
            x = self.layers[i](x)
            x = self.relu(x)
            if i < len(self.layers) - 2:  # Apply dropout except for last hidden layer
                x = self.dropout(x)
        
        # Output layer
        x = self.layers[-1](x)
        
        return x


class FraudDetectionClient(fl.client.Client):
    """Federated learning client for fraud detection."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        client_id: str,
        config: Dict[str, Any],
        ipfs_service: Optional[IPFSService] = None,
        blockchain_service: Optional[BlockchainService] = None
    ):
        """
        Initialize the federated client.
        
        Args:
            model: PyTorch model
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device to run the model on (CPU or GPU)
            client_id: Unique identifier for the client
            config: Configuration dictionary
            ipfs_service: IPFS service for model storage (optional)
            blockchain_service: Blockchain service for model verification (optional)
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.client_id = client_id
        self.config = config
        self.ipfs_service = ipfs_service
        self.blockchain_service = blockchain_service
        
        # Set up optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.get("learning_rate", 0.001)
        )
        
        # Metrics history
        self.metrics_history = []
        
        # Set up logging
        self.logger = self._setup_logger()
    
    def _setup_logger(self):
        """Set up logging."""
        import logging
        
        # Create logger
        logger = logging.getLogger(f"federated.client.{self.client_id}")
        logger.setLevel(logging.DEBUG)
        
        # Clear existing handlers
        if logger.handlers:
            logger.handlers.clear()
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
        # File handler
        log_dir = self.config.get("log_dir", "logs")
        os.makedirs(log_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(log_dir, f"client_{self.client_id}.log"))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
        return logger
    
    def get_properties(self, config: Config) -> GetPropertiesRes:
        """Return client properties."""
        properties = {
            "client_id": self.client_id,
            "address": self.config.get("blockchain", {}).get("account_address", "unknown"),
            "dataset_size": len(self.train_loader.dataset),
            "batch_size": self.train_loader.batch_size,
            "device": str(self.device)
        }
        
        self.logger.info(f"Returning properties: {properties}")
        return GetPropertiesRes(
            status=Status(code=Code.OK, message="Success"),
            properties=properties
        )
    
    def get_parameters(self, config: GetParametersIns) -> GetParametersRes:
        """Return model parameters as a list of numpy arrays."""
        self.logger.info("Getting model parameters")
        
        # Extract parameters from the model
        params = []
        for name, param in self.model.named_parameters():
            self.logger.debug(f"Parameter {name}: shape {param.shape}")
            params.append(param.cpu().detach().numpy())
        
        self.logger.info(f"Returning {len(params)} parameter tensors")
        return GetParametersRes(
            status=Status(code=Code.OK, message="Success"),
            parameters=ndarrays_to_parameters(params)
        )
    
    def fit(self, ins: FitIns) -> FitRes:
        """Train the model on the local dataset."""
        self.logger.info("Starting local training")
        
        # Get the current round and config
        config = ins.config
        current_round = config.get("round", 0)
        epochs = config.get("epochs", 1)
        batch_size = config.get("batch_size", 32)
        learning_rate = config.get("learning_rate", 0.001)
        global_model_hash = config.get("global_model_hash")
        
        self.logger.info(f"Fit config: round={current_round}, epochs={epochs}, lr={learning_rate}")
        
        # Load parameters from server
        parameters = ins.parameters
        params_array = parameters_to_ndarrays(parameters)
        
        # Log parameter shapes for debugging
        self.logger.info(f"Received {len(params_array)} parameter tensors")
        for i, param in enumerate(params_array):
            self.logger.debug(f"Parameter {i}: shape {param.shape}, dtype {param.dtype}")
        
        # Get model parameter shapes for comparison
        model_shapes = []
        for name, param in self.model.named_parameters():
            model_shapes.append((name, param.shape))
            
        self.logger.debug(f"Model expects parameters with shapes: {model_shapes}")
        
        # Check if parameter count matches
        if len(params_array) != len(list(self.model.parameters())):
            self.logger.warning(
                f"Parameter count mismatch: received {len(params_array)}, "
                f"model expects {len(list(self.model.parameters()))}"
            )
            
        # Try to load parameters safely
        try:
            params_dict = zip(self.model.state_dict().keys(), params_array)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            self.model.load_state_dict(state_dict)
            self.logger.info("Parameters successfully loaded into model")
        except Exception as e:
            self.logger.error(f"Error loading parameters: {e}")
            # Try an alternative loading method: directly assign to model parameters
            try:
                self.logger.info("Trying alternative parameter loading method")
                for param_tensor, param_model in zip(params_array, self.model.parameters()):
                    # Check shapes
                    if param_tensor.shape == param_model.shape:
                        param_model.data = torch.tensor(param_tensor)
                    else:
                        self.logger.warning(
                            f"Shape mismatch: parameter tensor {param_tensor.shape}, "
                            f"model parameter {param_model.shape}"
                        )
                self.logger.info("Parameters loaded using alternative method")
            except Exception as e2:
                self.logger.error(f"Alternative parameter loading also failed: {e2}")
                self.logger.warning("Using model with original parameters")
        
        # Update learning rate if provided
        if learning_rate is not None:
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = learning_rate
        
        # Train the model
        train_loss, train_accuracy = self._train(epochs)
        
        # Get updated model parameters
        parameters = [val.cpu().numpy() for _, val in self.model.state_dict().items()]
        
        # Metrics to report back to the server
        metrics = {
            "train_loss": float(train_loss),
            "train_accuracy": float(train_accuracy),
            "epoch_count": epochs,
            "dataset_size": len(self.train_loader.dataset)
        }
        
        # Store training metrics locally
        self.metrics_history.append({
            "round": current_round,
            "metrics": metrics,
            "timestamp": time.time()
        })
        
        self.logger.info(f"Training completed: {metrics}")
        
        # If IPFS is available, store the model
        ipfs_hash = None
        if self.ipfs_service:
            try:
                self.logger.info("Storing model on IPFS")
                ipfs_hash = self.ipfs_service.store_model_params(
                    parameters,
                    metadata={
                        "client_id": self.client_id,
                        "round": current_round,
                        "metrics": metrics,
                        "type": "client_update"
                    }
                )
                self.logger.info(f"Model stored on IPFS: {ipfs_hash}")
                
                # Add IPFS hash to metrics
                metrics["ipfs_hash"] = ipfs_hash
                
                # Submit to blockchain if configured
                if self.blockchain_service and "account_address" in self.config.get("blockchain", {}):
                    try:
                        metrics_json = json.dumps(metrics, default=str)
                        tx_hash = self.blockchain_service.submit_model_update(
                            current_round,
                            ipfs_hash,
                            metrics_json
                        )
                        self.logger.info(f"Submitted model update to blockchain: {tx_hash}")
                        metrics["tx_hash"] = tx_hash
                    except Exception as e:
                        self.logger.error(f"Failed to submit model update to blockchain: {e}")
            except Exception as e:
                self.logger.error(f"Failed to store model on IPFS: {e}")
        
        return FitRes(
            status=Status(code=Code.OK, message="Success"),
            parameters=ndarrays_to_parameters(parameters),
            num_examples=len(self.train_loader.dataset),
            metrics=metrics
        )
    
    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        """Evaluate the model on the local validation dataset."""
        self.logger.info("Starting evaluation")
        
        # Get config
        config = ins.config
        current_round = config.get("round", 0)
        global_model_hash = config.get("global_model_hash")
        
        # Check if we should load the model from IPFS
        if global_model_hash and self.ipfs_service:
            try:
                self.logger.info(f"Loading global model from IPFS for evaluation: {global_model_hash}")
                model_data = self.ipfs_service.retrieve_model_params(global_model_hash)
                
                # Update model parameters
                params_dict = zip(self.model.state_dict().keys(), model_data["params"])
                state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
                self.model.load_state_dict(state_dict)
                
                self.logger.info("Successfully loaded global model from IPFS")
            except Exception as e:
                self.logger.error(f"Failed to load model from IPFS: {e}")
        else:
            # Load parameters from server
            parameters = ins.parameters
            params_dict = zip(self.model.state_dict().keys(), parameters_to_ndarrays(parameters))
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            self.model.load_state_dict(state_dict)
            self.logger.info("Parameters received from server and loaded into model")
        
        # Evaluate
        loss, accuracy, metrics = self._evaluate()
        
        # Add to metrics
        metrics["loss"] = float(loss)
        metrics["accuracy"] = float(accuracy)
        
        self.logger.info(f"Evaluation completed: {metrics}")
        
        # If blockchain service is configured, submit evaluation
        if self.blockchain_service and "account_address" in self.config.get("blockchain", {}):
            try:
                self.logger.info(f"Submitting evaluation to blockchain for round {current_round}")
                tx_hash = self.blockchain_service.submit_evaluation(
                    current_round,
                    loss * 1000,  # Convert to uint256 (multiply by 1000)
                    accuracy * 1000,
                    metrics.get("auc", 0) * 1000,
                    metrics.get("precision", 0) * 1000,
                    metrics.get("recall", 0) * 1000
                )
                self.logger.info(f"Evaluation submitted to blockchain: {tx_hash}")
                metrics["tx_hash"] = tx_hash
            except Exception as e:
                self.logger.error(f"Failed to submit evaluation to blockchain: {e}")
        
        return EvaluateRes(
            status=Status(code=Code.OK, message="Success"),
            loss=float(loss),
            num_examples=len(self.val_loader.dataset),
            metrics=metrics
        )
    
    def _train(self, epochs: int) -> Tuple[float, float]:
        """
        Train the model for the given number of epochs.
        
        Returns:
            Tuple of (loss, accuracy)
        """
        self.model.train()
        self.model.to(self.device)
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        # Training loop
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0
            
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                # Reset gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                output = self.model(data)
                
                # Convert output to appropriate shape for binary classification
                output = output.squeeze()
                
                # Compute loss
                loss = F.binary_cross_entropy_with_logits(output, target.float())
                
                # Backward pass
                loss.backward()
                
                # Update parameters
                self.optimizer.step()
                
                # Update metrics
                epoch_loss += loss.item() * data.size(0)
                
                # Compute accuracy
                pred = torch.sigmoid(output) >= 0.5
                epoch_correct += (pred == target).sum().item()
                epoch_total += target.size(0)
                
                if batch_idx % 10 == 0:
                    self.logger.debug(f"Epoch {epoch}/{epochs}, Batch {batch_idx}: Loss {loss.item():.4f}")
            
            # Epoch statistics
            epoch_loss = epoch_loss / epoch_total
            epoch_accuracy = epoch_correct / epoch_total
            
            self.logger.info(f"Epoch {epoch}/{epochs}: Loss {epoch_loss:.4f}, Accuracy {epoch_accuracy:.4f}")
            
            total_loss += epoch_loss
            correct += epoch_correct
            total += epoch_total
        
        # Average over epochs
        avg_loss = total_loss / epochs
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def _evaluate(self) -> Tuple[float, float, Dict[str, float]]:
        """
        Evaluate the model on the validation set.
        
        Returns:
            Tuple of (loss, accuracy, additional_metrics)
        """
        self.model.eval()
        self.model.to(self.device)
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        # For ROC AUC calculation
        all_targets = []
        all_preds = []
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                # Forward pass
                output = self.model(data)
                output = output.squeeze()
                
                # Compute loss
                loss = F.binary_cross_entropy_with_logits(output, target.float())
                
                # Update metrics
                total_loss += loss.item() * data.size(0)
                
                # Compute predictions
                pred_probs = torch.sigmoid(output)
                pred = pred_probs >= 0.5
                correct += (pred == target).sum().item()
                total += target.size(0)
                
                # Store for additional metrics
                all_targets.extend(target.cpu().numpy())
                all_preds.extend(pred_probs.cpu().numpy())
        
        # Calculate metrics
        loss = total_loss / total
        accuracy = correct / total
        
        # Additional metrics (precision, recall, F1, AUC)
        metrics = self._compute_additional_metrics(np.array(all_targets), np.array(all_preds))
        
        return loss, accuracy, metrics
    
    def _compute_additional_metrics(self, targets: np.ndarray, pred_probs: np.ndarray) -> Dict[str, float]:
        """Compute additional evaluation metrics."""
        from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
        
        # Convert probabilities to binary predictions
        predictions = (pred_probs >= 0.5).astype(int)
        
        try:
            precision = precision_score(targets, predictions, zero_division=0)
            recall = recall_score(targets, predictions, zero_division=0)
            f1 = f1_score(targets, predictions, zero_division=0)
            
            # ROC AUC (using probabilities)
            auc = roc_auc_score(targets, pred_probs)
            
            return {
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1),
                "auc": float(auc)
            }
        except Exception as e:
            self.logger.error(f"Error computing metrics: {e}")
            return {}


def load_data(config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
    """
    Load the fraud detection dataset.
    
    This function should be customized based on your specific dataset.
    
    Args:
        config: Configuration dictionary with dataset parameters
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # This is a placeholder - implement actual data loading based on your dataset
    # For example:
    
    class DummyDataset(Dataset):
        """Dummy dataset for testing."""
        def __init__(self, size=1000, input_dim=50, seed=42):
            np.random.seed(seed)
            self.data = np.random.randn(size, input_dim).astype(np.float32)
            self.labels = np.random.randint(0, 2, size=size).astype(np.float32)
            
        def __len__(self):
            return len(self.data)
            
        def __getitem__(self, idx):
            return self.data[idx], self.labels[idx]
    
    # Create datasets
    input_dim = config.get("input_dim", 50)
    train_size = config.get("train_size", 800)
    val_size = config.get("val_size", 200)
    batch_size = config.get("batch_size", 32)
    
    train_dataset = DummyDataset(size=train_size, input_dim=input_dim, seed=42)
    val_dataset = DummyDataset(size=val_size, input_dim=input_dim, seed=43)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_loader, val_loader


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Federated Fraud Detection Client")
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
        "--client-id", 
        type=str, 
        default=None,
        help="Unique client identifier (default: generated from hostname)"
    )
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from a JSON file."""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading config: {e}")
        return {}


def main():
    """Main function to run the client."""
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create log directory if not exists
    log_dir = config.get("log_dir", "client_logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # Setup basic logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(log_dir, "client.log"))
        ]
    )
    logger = logging.getLogger("federated.client")
    
    # Generate client ID if not provided
    if args.client_id is None:
        import socket
        hostname = socket.gethostname()
        args.client_id = f"{hostname}_{int(time.time())}"
    
    logger.info(f"Starting client with ID: {args.client_id}")
    
    # Setup device (CPU or GPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    try:
        # Load data
        logger.info("Loading datasets...")
        train_loader, val_loader = load_data(config)
        logger.info(f"Loaded training set with {len(train_loader.dataset)} samples")
        logger.info(f"Loaded validation set with {len(val_loader.dataset)} samples")
        
        # Create model
        input_dim = config.get("input_dim", 50)
        hidden_dims = config.get("hidden_dims", [32, 16])
        model = FraudDetectionModel(input_dim, hidden_dims)
        logger.info(f"Created model with input dimension {input_dim}")
        
        # Create blockchain service if configured
        blockchain_service = None
        if "blockchain" in config:
            try:
                blockchain_service = BlockchainService(
                    provider_url=config["blockchain"]["provider_url"],
                    contract_address=config["blockchain"]["contract_address"],
                    contract_abi_path=config["blockchain"]["contract_abi_path"],
                    private_key=config["blockchain"].get("private_key"),
                    account_address=config["blockchain"].get("account_address")
                )
                logger.info(f"Blockchain service initialized with address: {config['blockchain'].get('account_address', 'unknown')}")
            except Exception as e:
                logger.error(f"Failed to initialize blockchain service: {e}")
                logger.info("Continuing without blockchain integration")
        
        # Create IPFS service if configured
        ipfs_service = None
        if "ipfs" in config:
            try:
                ipfs_service = IPFSService(
                    api_url=config["ipfs"]["api_url"]
                )
                logger.info(f"IPFS service initialized with API URL: {config['ipfs']['api_url']}")
            except Exception as e:
                logger.error(f"Failed to initialize IPFS service: {e}")
                logger.info("Continuing without IPFS integration")
        
        # Create client
        client = FraudDetectionClient(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            client_id=args.client_id,
            config=config,
            ipfs_service=ipfs_service,
            blockchain_service=blockchain_service
        )
        
        # Start client
        logger.info(f"Starting Flower client, connecting to server at {args.server_address}")
        fl.client.start_client(
            server_address=args.server_address,
            client=client
        )
        
    except Exception as e:
        logger.error(f"Error in client execution: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    main()