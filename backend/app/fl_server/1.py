import flwr as fl
import argparse
from typing import List, Tuple, Dict, Optional
from flwr.common import Metrics

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate accuracy using weighted average."""
    # Calculate the total number of examples used
    examples = [num_examples for num_examples, _ in metrics]
    total_examples = sum(examples)
    
    # Create a weighted accuracy metric
    weighted_metrics = {}
    
    # For each metric provided by the clients
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
        "epochs": 1,  # Number of local epochs
        "batch_size": 32,
        "learning_rate": 0.01,
        "current_round": server_round,
    }
    return config

def evaluate_config(server_round: int) -> Dict:
    """Return evaluation configuration dict for each round."""
    return {"round": server_round}

def main():
    parser = argparse.ArgumentParser(description="Flower Server")
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

    # Define the server strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,  # Sample 100% of available clients for training
        fraction_evaluate=1.0,  # Sample 100% available clients for evaluation
        min_fit_clients=args.min_clients,  # Never sample less than min_clients
        min_evaluate_clients=args.min_clients,  # Never sample less than min_clients
        min_available_clients=args.min_available_clients,  # Wait until min_available_clients are available
        evaluate_metrics_aggregation_fn=weighted_average,  # Custom aggregation function
        on_fit_config_fn=fit_config,  # Function for fit configuration
        on_evaluate_config_fn=evaluate_config,  # Function for evaluate configuration
    )

    # Start the server
    print(f"Starting Flower server at {args.server_address}")
    print(f"Expecting minimum {args.min_clients} clients to join")
    fl.server.start_server(
        server_address=args.server_address,
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()
