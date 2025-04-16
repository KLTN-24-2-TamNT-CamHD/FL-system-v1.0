import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import argparse
import logging
from collections import OrderedDict
from typing import Dict, List, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("federated.client")

# Define a simple model
class SimpleModel(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super(SimpleModel, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            prev_dim = dim
        
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

# Define a simple synthetic dataset
class SyntheticDataset(Dataset):
    def __init__(self, num_samples=1000, input_dim=50, noise=0.1):
        self.input_dim = input_dim
        self.num_samples = num_samples
        
        # Generate random weights for the true model
        self.true_weights = torch.randn(input_dim, 1)
        
        # Generate random input data
        self.X = torch.randn(num_samples, input_dim)
        
        # Generate output with some noise
        self.y = self.X @ self.true_weights + noise * torch.randn(num_samples, 1)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Flower client
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, val_loader, client_id):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.client_id = client_id
        self.logger = logging.getLogger(f"federated.client.{client_id}")
    
    def get_parameters(self, config):
        self.logger.info("Getting model parameters")
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def set_parameters(self, parameters):
        self.logger.info("Setting model parameters")
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)
    
    def fit(self, parameters, config):
        self.logger.info(f"Training in round {config.get('current_round', 0)}")
        self.set_parameters(parameters)
        
        # Get training configuration
        epochs = config.get("epochs", 1)
        batch_size = config.get("batch_size", 32)
        learning_rate = config.get("learning_rate", 0.01)
        
        # Define loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)
        
        # Training loop
        train_loss = 0.0
        examples = 0
        self.model.train()
        for _ in range(epochs):
            for batch_idx, (data, target) in enumerate(self.train_loader):
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * data.size(0)
                examples += data.size(0)
        
        # Return updated model parameters and training statistics
        return self.get_parameters({}), examples, {"loss": train_loss / examples}
    
    def evaluate(self, parameters, config):
        self.logger.info(f"Evaluating in round {config.get('round', 0)}")
        self.set_parameters(parameters)
        
        # Define loss function
        criterion = nn.MSELoss()
        loss = 0.0
        examples = 0
        
        # Evaluation loop
        self.model.eval()
        with torch.no_grad():
            for data, target in self.val_loader:
                output = self.model(data)
                loss += criterion(output, target).item() * data.size(0)
                examples += data.size(0)
        
        return loss, examples, {"loss": loss / examples}

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Flower client")
    parser.add_argument(
        "--server-address",
        type=str,
        default="127.0.0.1:8088",
        help="Server address (IP:port)"
    )
    parser.add_argument(
        "--client-id",
        type=str,
        required=True,
        help="Client ID"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a config file"
    )
    args = parser.parse_args()
    
    # Print client information
    logger.info(f"Starting client with ID: {args.client_id}")
    
    # Determine device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create datasets
    logger.info("Loading datasets...")
    dataset = SyntheticDataset(num_samples=1000, input_dim=50)
    train_dataset, val_dataset = random_split(dataset, [800, 200])
    logger.info(f"Loaded training set with {len(train_dataset)} samples")
    logger.info(f"Loaded validation set with {len(val_dataset)} samples")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    # Create model
    input_dim = 50
    hidden_dims = [32, 16]
    model = SimpleModel(input_dim, hidden_dims)
    logger.info(f"Created model with input dimension {input_dim} and hidden dimensions {hidden_dims}")
    
    # Start Flower client
    client = FlowerClient(model, train_loader, val_loader, args.client_id)
    logger.info(f"Starting Flower client, connecting to server at {args.server_address}")
    fl.client.start_numpy_client(server_address=args.server_address, client=client)

if __name__ == "__main__":
    main()
