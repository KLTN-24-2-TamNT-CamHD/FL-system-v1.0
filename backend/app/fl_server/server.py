from typing import Dict, List, Optional, Tuple
import flwr as fl
from flwr.common import Metrics
from flwr.server.client_proxy import ClientProxy
import numpy as np
from app.services import blockchain_service
import asyncio
from datetime import datetime

class FraudDetectionServer(fl.server.strategy.FedAvg):
    def __init__(
        self,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_metrics_aggregation_fn: Optional[callable] = None,
    ) -> None:
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        )
        self.current_round = 0
        print(f"Flower Server initialized at {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Server admin: dinhcam89")

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, fl.common.FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[fl.common.Parameters], Dict[str, float]]:
        # Aggregate model updates
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if aggregated_parameters is not None:
            # Store aggregated model in IPFS and get hash
            # Note: You'll need to implement the actual IPFS storage
            model_hash = f"QmHash{server_round}"  # Placeholder
            
            # Update blockchain with new global model
            asyncio.get_event_loop().run_until_complete(
                blockchain_service.complete_round(server_round, model_hash)
            )
            
            print(f"Round {server_round} completed - Model hash: {model_hash}")
            print(f"Timestamp (UTC): 2025-03-31 15:01:55")
            print(f"Admin: dinhcam89")

        return aggregated_parameters, aggregated_metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, fl.common.EvaluateRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[float], Dict[str, float]]:
        """Aggregate evaluation metrics."""
        if not results:
            return None, {}

        # Aggregate metrics
        loss_aggregated = sum(r.metrics["loss"] * r.num_examples for _, r in results) / sum(
            r.num_examples for _, r in results
        )
        
        metrics_aggregated = {
            "loss": int(loss_aggregated),
            "accuracy": int(np.mean([r.metrics["accuracy"] for _, r in results]) * 100),
            "auc": int(np.mean([r.metrics.get("auc", 0) for _, r in results]) * 100),
            "precision": int(np.mean([r.metrics.get("precision", 0) for _, r in results]) * 100),
            "recall": int(np.mean([r.metrics.get("recall", 0) for _, r in results]) * 100),
        }

        # Submit evaluation metrics to blockchain
        asyncio.get_event_loop().run_until_complete(
            blockchain_service.submit_evaluation(
                server_round,
                metrics_aggregated["loss"],
                metrics_aggregated["accuracy"],
                metrics_aggregated["auc"],
                metrics_aggregated["precision"],
                metrics_aggregated["recall"]
            )
        )

        print(f"Round {server_round} evaluation - Metrics: {metrics_aggregated}")
        return loss_aggregated, metrics_aggregated

def start_server(port: int = 8080):
    strategy = FraudDetectionServer(
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2
    )

    # Start Flower server
    fl.server.start_server(
        server_address=f"0.0.0.0:{port}",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
    )
