import flwr as fl
import argparse
import logging
import torch
import torch.nn as nn
import os
from typing import List, Tuple, Dict, Optional, Any
from flwr.common import Metrics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("federated.server")

# Simple model for testing
class SimpleModel(nn.Module):
    def __init__(self, input_dim=50, hidden_dims=[32, 16]):
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

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate accuracy using weighted average."""
    examples = [num_examples for num_examples, _ in metrics]
    total_examples = sum(examples)
    
    weighted_metrics = {}
    
    if total_examples == 0:
        return {}
    
    for metric_name in metrics[0][1].keys():
        weighted_metrics[metric_name] = sum(
            num_examples * m[metric_name] for num_examples, m in metrics
        ) / total_examples
    
    return weighted_metrics

def fit_config(server_round: int) -> Dict:
    """Return a configuration with static batch size and (local) epochs."""
    config = {
        "epochs": 1,
        "batch_size": 32,
        "learning_rate": 0.01,
        "current_round": server_round,
    }
    return config

def evaluate_config(server_round: int) -> Dict:
    """Return evaluation configuration dict for each round."""
    return {"round": server_round}

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Simplified Flower Server")
    parser.add_argument(
        "--min-clients", 
        type=int, 
        default=2, 
        help="Minimum number of clients to start training"
    )
    parser.add_argument(
        "--min-available-clients", 
        type=int, 
        default=2, 
        help="Minimum number of available clients required"
    )
    parser.add_argument(
        "--num-rounds", 
        type=int, 
        default=3, 
        help="Number of rounds of federated learning"
    )
    parser.add_argument(
        "--server-address",
        type=str,
        default="0.0.0.0:8088",
        help="Server address (IP:port)"
    )
    args = parser.parse_args()
    
    # Initialize model
    model = SimpleModel(input_dim=50, hidden_dims=[32, 16])
    
    # Get initial parameters
    try:
        initial_parameters = [val.cpu().numpy() for _, val in model.state_dict().items()]
        
        # Check if the Flower version has ndarrays_to_parameters
        if hasattr(fl.common, "ndarrays_to_parameters"):
            # For newer Flower versions
            initial_parameters = fl.common.ndarrays_to_parameters(initial_parameters)
        else:
            # For older Flower versions
            initial_parameters = fl.common.weights_to_parameters(initial_parameters)
    except Exception as e:
        logger.error(f"Error preparing initial parameters: {e}")
        return
        
    # Define strategy with appropriate parameters method
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=args.min_clients,
        min_evaluate_clients=args.min_clients,
        min_available_clients=args.min_available_clients,
        evaluate_metrics_aggregation_fn=weighted_average,
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        initial_parameters=initial_parameters,
    )

    # Start server
    logger.info(f"Starting Flower server at {args.server_address}")
    logger.info(f"Expecting minimum {args.min_clients} clients to join")
    fl.server.start_server(
        server_address=args.server_address,
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()
