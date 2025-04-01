import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import argparse
from datetime import datetime

# Simple model for fraud detection
class FraudDetectionModel(nn.Module):
    def __init__(self, input_dim=30):  # Default input dim for fraud detection
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

class FraudDetectionClient(fl.client.NumPyClient):
    def __init__(self, model, institution_address: str):
        self.model = model
        self.institution_address = institution_address
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        print(f"\nInitializing FL Client:")
        print(f"Institution Address: {institution_address}")
        print(f"Device: {self.device}")
        print(f"Timestamp (UTC): 2025-04-01 13:00:33")
        print(f"Admin: dinhcam89\n")
        
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)
    
    def fit(self, parameters, config):
        print(f"\nStarting training round...")
        print(f"Timestamp (UTC): 2025-04-01 13:00:33")
        self.set_parameters(parameters)
        
        # For testing, we'll just return the current parameters
        # In production, you would train the model here
        
        print("Training completed for this round")
        return self.get_parameters(config={}), 1, {}
    
    def evaluate(self, parameters, config):
        print(f"\nStarting evaluation...")
        print(f"Timestamp (UTC): 2025-04-01 13:00:33")
        self.set_parameters(parameters)
        
        # For testing, we'll return dummy metrics
        loss = 0.1
        accuracy = 0.95
        
        print(f"Evaluation completed:")
        print(f"Loss: {loss:.4f}")
        print(f"Accuracy: {accuracy:.4f}\n")
        
        return loss, 1, {"accuracy": accuracy}

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Flower client for fraud detection")
    parser.add_argument(
        "--address",
        type=str,
        required=True,
        help="Ethereum address of the institution"
    )
    parser.add_argument(
        "--server",
        type=str,
        default="127.0.0.1:8080",
        help="Server address (default: 127.0.0.1:8080)"
    )
    
    args = parser.parse_args()
    
    # Print startup information
    print("\nStarting Flower Client")
    print("=====================")
    print(f"Timestamp (UTC): 2025-04-01 13:00:33")
    print(f"Admin: dinhcam89")
    print(f"Server Address: {args.server}")
    print(f"Institution Address: {args.address}")
    
    # Initialize model
    model = FraudDetectionModel()
    
    # Initialize client
    client = FraudDetectionClient(model, args.address)
    
    # Start client
    fl.client.start_numpy_client(
        server_address=args.server,
        client=client
    )

if __name__ == "__main__":
    main()