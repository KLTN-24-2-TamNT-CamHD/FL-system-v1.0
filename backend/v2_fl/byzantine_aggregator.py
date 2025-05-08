# Byzantine-Robust Federated Learning Integration Guide

## Overview

This guide explains how to integrate the Byzantine-robust methods into your existing federated learning system to defend against malicious clients submitting low-performance model parameters.

## 1. Replace Your Current Aggregator

Replace your current `EnsembleAggregator` with the new `ByzantineRobustEnsembleAggregator`:

```python
from byzantine_robust_aggregation import ByzantineRobustEnsembleAggregator

# Initialize with desired robustness settings
aggregator = ByzantineRobustEnsembleAggregator(
    device="cpu",
    robust_method="trimmed_mean",  # Options: "trimmed_mean", "median", "krum"
    trim_ratio=0.2,                # For trimmed_mean: ratio of models to trim (0.0-0.5)
    z_score_threshold=2.0,         # Threshold for outlier detection
    min_client_contribution=3,     # Minimum client models needed for aggregation
    reputation_threshold=0.6       # Minimum reputation score for trusted clients
)
```

## 2. Update Your Training Strategy

Modify your Flower `Strategy` class to use the new aggregator:

```python
from flwr.server.strategy import FedAvg

class RobustFederatedStrategy(FedAvg):
    def __init__(
        self,
        *args,
        robust_aggregator=None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.robust_aggregator = robust_aggregator if robust_aggregator else ByzantineRobustEnsembleAggregator()

    def aggregate_fit(self, server_round, results, failures):
        """Aggregate model weights using robust methods."""
        if not results:
            return None, {}

        # Extract weights from results
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        weights = [w[0] for w in weights_results]
        examples = [w[1] for w in weights_results]

        # Use robust aggregator instead of standard federated averaging
        return self.robust_aggregator.aggregate_fit_results(results, examples)
        
    def aggregate_evaluate(self, server_round, results, failures):
        """Aggregate evaluation metrics using robust methods."""
        if not results:
            return None
        
        # Use robust aggregator for evaluation metrics
        return self.robust_aggregator.aggregate_evaluate_results(
            results, [r.num_examples for _, r in results]
        )
```

## 3. Configure Your Federated Server

Update your server initialization code:

```python
# Initialize the robust aggregator
robust_aggregator = ByzantineRobustEnsembleAggregator(
    device="cpu",
    robust_method="trimmed_mean",
    trim_ratio=0.2
)

# Create strategy with the robust aggregator
strategy = RobustFederatedStrategy(
    robust_aggregator=robust_aggregator,
    # Your other Flower strategy parameters...
    min_fit_clients=5,
    min_available_clients=5,
    fraction_fit=1.0
)

# Start Flower server with robust strategy
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=10),
    strategy=strategy
)
```

## 4. Implement Client-Side Validation (Optional)

To further enhance security, implement validation on the client side:

```python
class ByzantineRobustClient(fl.client.NumPyClient):
    def __init__(self, model, train_data, validation_data):
        self.model = model
        self.train_data = train_data
        self.validation_data = validation_data
        
    def fit(self, parameters, config):
        # Standard training process
        # ...
        
        # Validate local model before sending
        val_loss, val_accuracy = self.model.evaluate(self.validation_data)
        
        # Only send model if it meets quality threshold
        # This helps prevent accidental poor-quality models
        if val_accuracy < 0.2:  # Adjust based on your application
            logging.warning("Local model has poor performance, retraining...")
            # Retry training or use previous model
            
        return self.model.get_weights(), len(self.train_data), {}
```

## 5. Monitoring Byzantine Defense

Add monitoring code to track defense performance:

```python
# In your server code, after each round
def on_round_completed(server_round, results, failures):
    """Process the results of each round."""
    # Get metrics from aggregator
    num_trusted = strategy.robust_aggregator.metrics.get("num_trusted_ensembles", 0)
    num_total = strategy.robust_aggregator.metrics.get("num_ensembles", 0)
    
    # Log defense statistics
    logging.info(f"Round {server_round}: {num_trusted}/{num_total} clients trusted")
    
    # Log client reputations
    for client_id, reputation in strategy.robust_aggregator.client_reputations.items():
        logging.info(f"Client {client_id} reputation: {reputation:.2f}")
    
    # Optional: Save defense stats to file for later analysis
    with open(f"defense_stats_round_{server_round}.json", "w") as f:
        json.dump(strategy.robust_aggregator.metrics, f)
```

## 6. Testing Byzantine Robustness (Recommended)

Create test cases to verify the robustness of your system:

```python
def test_byzantine_robustness():
    """Test the system's robustness against malicious clients."""
    # 1. Set up normal clients
    normal_clients = [create_client(i, malicious=False) for i in range(8)]
    
    # 2. Create malicious clients with different attack patterns
    malicious_clients = [
        create_client(8, malicious=True, attack="random"),     # Random parameters
        create_client(9, malicious=True, attack="flip_sign"),  # Flip signs of gradients
        create_client(10, malicious=True, attack="boost"),     # Boost parameter values
    ]
    
    all_clients = normal_clients + malicious_clients
    
    # 3. Run federated learning with mixed clients
    results_with_defense = run_federated_learning(
        all_clients, 
        use_robust_aggregation=True,
        rounds=5
    )
    
    # 4. Run without defenses as baseline
    results_without_defense = run_federated_learning(
        all_clients, 
        use_robust_aggregation=False,
        rounds=5
    )
    
    # 5. Compare performance
    plot_performance_comparison(results_with_defense, results_without_defense)
```

## 7. Blockchain Integration (For Your IPFS/Blockchain System)

Update your blockchain connector to store reputation scores:

```python
def save_model_with_reputation(self, ipfs_hash, client_id, round_num, reputation_score):
    """Save model to blockchain with reputation information."""
    try:
        tx_hash = self.contract.functions.addClientModel(
            self._to_bytes32(ipfs_hash),
            client_id,
            round_num,
            int(reputation_score * 100),  # Convert to 0-100 scale
            self._to_bytes32("")
        ).transact({'from': self.account_address})
        
        return self.w3.eth.wait_for_transaction_receipt(tx_hash)
    except Exception as e:
        logging.error(f"Error saving model with reputation: {e}")
        return None
```

## Best Practices for Byzantine-Robust Federated Learning

1. **Start conservative**: Begin with a high reputation threshold and gradually lower it as needed.

2. **Use multiple defense mechanisms**: Combine statistical outlier detection, reputation tracking, and robust aggregation for best results.

3. **Monitor client behavior**: Track client reputations over time to identify patterns of malicious behavior.

4. **Adjust thresholds dynamically**: As your system matures, adjust detection thresholds based on observed data.

5. **Maintain a validation dataset**: A high-quality central validation dataset is essential for verifying client contributions.

6. **Implement graceful degradation**: Even if many clients are flagged as suspicious, your system should continue functioning.

7. **Balance security and inclusion**: Overly strict Byzantine defenses might exclude legitimate clients with unusual but valuable data.