import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from fl_server.client import FraudDetectionClient
import flwr as fl
from datetime import datetime

# Simple model for testing
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.linear(x))

def main(institution_address: str):
    # Initialize model and data loaders
    model = SimpleModel()
    
    # Create dummy data loaders for testing
    train_loader = [(torch.randn(32, 10), torch.randint(0, 2, (32,))) for _ in range(10)]
    val_loader = [(torch.randn(32, 10), torch.randint(0, 2, (32,))) for _ in range(5)]
    
    # Create client
    client = FraudDetectionClient(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        institution_address=institution_address
    )
    
    # Start client
    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=client
    )

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python start_fl_client.py <institution_address>")
        sys.exit(1)
        
    institution_address = sys.argv[1]
    print(f"Starting Flower client for institution: {institution_address}")
    print(f"Timestamp (UTC): 2025-03-31 15:01:55")
    print(f"Admin: dinhcam89")
    main(institution_address)