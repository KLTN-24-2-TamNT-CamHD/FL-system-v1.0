"""
Enhanced Federated Learning Server with IPFS and Blockchain integration.
Supports client authorization and contribution tracking using wallet addresses.
"""

import os
import json
import pickle
import time
from typing import Dict, List, Optional, Tuple, Union, Any, Set
from datetime import datetime, timezone
import logging
from pathlib import Path

import flwr as fl
from flwr.server.client_proxy import ClientProxy
from flwr.common import (
    Parameters,
    Scalar,
    FitIns,
    FitRes,
    EvaluateIns,
    EvaluateRes,
    parameters_to_ndarrays,
    ndarrays_to_parameters,
)
import numpy as np

from ipfs_connector import IPFSConnector
from blockchain_connector import BlockchainConnector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("FL-Server")

class EnhancedFedAvg(fl.server.strategy.FedAvg):
    """Enhanced Federated Averaging strategy with IPFS, blockchain authentication and rewards."""
    
    def __init__(
        self,
        *args,
        ipfs_connector: Optional[IPFSConnector] = None,
        blockchain_connector: Optional[BlockchainConnector] = None,
        version_prefix: str = "1.0",
        authorized_clients_only: bool = True,
        round_rewards: int = 1000,  # Reward points to distribute each round
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        
        # Initialize IPFS connector
        self.ipfs = ipfs_connector or IPFSConnector()
        
        # Initialize blockchain connector
        self.blockchain = blockchain_connector
        
        # Set version prefix
        self.version_prefix = version_prefix
        
        # Flag to only allow authorized clients
        self.authorized_clients_only = authorized_clients_only
        
        # Rewards per round
        self.round_rewards = round_rewards
        
        # Set of authorized clients from blockchain
        self.authorized_clients: Set[str] = set()
        
        # Client contributions for current round
        self.current_round_contributions = {}
        
        # Metrics storage
        self.metrics_history = []
        
        # Load authorized clients from blockchain
        self._load_authorized_clients()
        
        logger.info(f"Initialized EnhancedFedAvg with IPFS node: {self.ipfs.ipfs_api_url}")
        if self.blockchain:
            logger.info(f"Blockchain integration enabled")
            if self.authorized_clients_only:
                logger.info(f"Only accepting contributions from authorized clients ({len(self.authorized_clients)} loaded)")
    
    def _load_authorized_clients(self):
        """Load authorized clients from blockchain."""
        if self.blockchain:
            try:
                clients = self.blockchain.get_all_authorized_clients()
                self.authorized_clients = set(clients)
                logger.info(f"Loaded {len(self.authorized_clients)} authorized clients from blockchain")
            except Exception as e:
                logger.error(f"Failed to load authorized clients: {e}")
    
    def is_client_authorized(self, wallet_address: str) -> bool:
        """Check if a client is authorized."""
        if not self.authorized_clients_only:
            return True
        
        if not wallet_address or wallet_address == "unknown":
            logger.warning(f"Client provided no wallet address")
            return False
        
        # Check local cache first
        if wallet_address in self.authorized_clients:
            return True
        
        # Check blockchain
        if self.blockchain:
            try:
                is_authorized = self.blockchain.is_client_authorized(wallet_address)
                # Update local cache
                if is_authorized:
                    self.authorized_clients.add(wallet_address)
                
                return is_authorized
            except Exception as e:
                logger.error(f"Failed to check client authorization: {e}")
                # Fall back to local cache
                return wallet_address in self.authorized_clients
        
        return False
    
    def get_version(self, round_num: int) -> str:
        """Generate a version string based on round number."""
        return f"{self.version_prefix}.{round_num}"
    
    def initialize_parameters(
        self, client_manager: fl.server.client_manager.ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        
        # Check if we already have a model in blockchain/IPFS
        if self.blockchain:
            try:
                # Try to get the latest model from blockchain
                latest_model = self.blockchain.get_latest_model(self.version_prefix)
                ipfs_hash = latest_model["ipfs_hash"]
                
                logger.info(f"Found existing model in blockchain: {ipfs_hash} (round {latest_model['round']})")
                
                # Retrieve model from IPFS
                model_data = self.ipfs.get_json(ipfs_hash)
                if model_data and "state_dict" in model_data:
                    # Convert state_dict to ndarrays
                    weights = [
                        np.array(v, dtype=np.float32)
                        for k, v in model_data["state_dict"].items()
                    ]
                    parameters = ndarrays_to_parameters(weights)
                    return parameters
            
            except Exception as e:
                logger.warning(f"Could not retrieve latest model from blockchain: {e}")
                logger.info("Falling back to random initialization")
        
        # Fall back to default initialization
        return super().initialize_parameters(client_manager)
    
    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: fl.server.client_manager.ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training, sending only IPFS hash."""
        
        # Reset round contributions
        self.current_round_contributions = {}
        
        # Store global model in IPFS
        weights = parameters_to_ndarrays(parameters)
        
        # Create state dict from weights
        state_dict = {}
        layer_names = ["linear.weight", "linear.bias"]  # Adjust based on your model
        
        for i, name in enumerate(layer_names):
            if i < len(weights):
                state_dict[name] = weights[i].tolist()
        
        # Create metadata
        model_metadata = {
            "state_dict": state_dict,
            "info": {
                "round": server_round,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "version": self.get_version(server_round)
            }
        }
        
        # Store in IPFS
        ipfs_hash = self.ipfs.add_json(model_metadata)
        logger.info(f"Stored global model in IPFS: {ipfs_hash}")
        
        # Register in blockchain if available
        if self.blockchain:
            try:
                # Use register_or_update_model instead of register_model
                tx_hash = self.blockchain.register_or_update_model(
                    ipfs_hash=ipfs_hash,
                    round_num=server_round,
                    version=self.get_version(server_round),
                    participating_clients=0  # Will be updated after fit
                )
                logger.info(f"Registered global model in blockchain, tx: {tx_hash}")
            except Exception as e:
                logger.error(f"Failed to register model in blockchain: {e}")
        
        # Include only IPFS hash in config
        #config = {"ipfs_hash": ipfs_hash, "server_round": server_round}
        
        # Create config for only transferring IPFS hash
        config = {
            "ipfs_hash": ipfs_hash, 
            "server_round": server_round,
            "cid_only": True  # Flag to indicate we're only sending CID
        }
        # Create empty/minimal parameters object
        minimal_params = ndarrays_to_parameters([np.array([0.0])])
        
        # Configure fit instructions for clients
        fit_ins = FitIns(minimal_params, config)
        
        # Sample clients for this round
        clients = client_manager.sample(
            num_clients=self.min_fit_clients, 
            min_num_clients=self.min_available_clients
        )
        
        # Return client/config pairs
        return [(client, fit_ins) for client in clients]
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate model updates from clients."""
        
        # Filter out unauthorized clients
        authorized_results = []
        unauthorized_clients = []
        rejected_models = []
        
        for client, fit_res in results:
            wallet_address = fit_res.metrics.get("wallet_address", "unknown")
            client_ipfs_hash = fit_res.metrics.get("client_ipfs_hash")
            
            # Check for auth error flag
            if fit_res.metrics.get("error") == "client_not_authorized":
                logger.warning(f"Client {wallet_address} reported as unauthorized")
                unauthorized_clients.append((client, wallet_address))
                continue
            
            # Verify client authorization
            if not self.is_client_authorized(wallet_address):
                logger.warning(f"Rejecting contribution from unauthorized client: {wallet_address}")
                unauthorized_clients.append((client, wallet_address))
                continue

            # Verify model integrity by checking against blockchain
            if self.blockchain and client_ipfs_hash:
                try:
                    # Get the client's contributions from blockchain
                    records = self.blockchain.get_client_contribution_records(wallet_address)
                    if not records or "ipfs_hashes" not in records:
                        logger.warning(f"No contribution records found for client {wallet_address}")
                        rejected_models.append((client, wallet_address, "no_records"))
                        continue
                    
                    # Check if the submitted IPFS hash matches what's on the blockchain
                    ipfs_hashes = records["ipfs_hashes"]
                    rounds = records["rounds"]
                    
                    # Find records for the current round
                    for i, r in enumerate(rounds):
                        if r == server_round and ipfs_hashes[i] == client_ipfs_hash:
                            # Hash verified on blockchain
                            logger.info(f"Verified model hash {client_ipfs_hash} on blockchain for client {wallet_address}")
                            authorized_results.append((client, fit_res))
                            
                            # Record contribution metrics
                            if client_ipfs_hash and wallet_address != "unknown":
                                accuracy = fit_res.metrics.get("accuracy", 0.0)
                                self.current_round_contributions[wallet_address] = {
                                    "ipfs_hash": client_ipfs_hash,
                                    "accuracy": accuracy
                                }
                            break
                    else:
                        # Hash not found on blockchain for this round
                        logger.warning(f"Rejecting model from {wallet_address}: Hash {client_ipfs_hash} not verified on blockchain")
                        rejected_models.append((client, wallet_address, "hash_mismatch"))
                    
                except Exception as e:
                    logger.error(f"Error verifying contribution on blockchain: {e}")
                    # In case of verification error, reject to be safe
                    rejected_models.append((client, wallet_address, "verification_error"))
            else:
                # If blockchain verification isn't available, accept the model but log a warning
                logger.warning(f"Accepting unverified model from {wallet_address} without blockchain verification")
                authorized_results.append((client, fit_res))
                
                # Record contribution metrics
                if client_ipfs_hash and wallet_address != "unknown":
                    accuracy = fit_res.metrics.get("accuracy", 0.0)
                    self.current_round_contributions[wallet_address] = {
                        "ipfs_hash": client_ipfs_hash,
                        "accuracy": accuracy
                    }
        
        # Check if enough clients returned results
        if not authorized_results:
            if unauthorized_clients:
                logger.error(f"All {len(unauthorized_clients)} clients were unauthorized. No aggregation possible.")
            else:
                logger.error("No clients returned results. No aggregation possible.")
            return None, {"error": "no_authorized_clients"}
        
        # Perform aggregation with only authorized clients
        parameters_aggregated, metrics = super().aggregate_fit(server_round, authorized_results, failures)
        
        if parameters_aggregated is not None:
            # Add metrics about client participation
            metrics["total_clients"] = len(results)
            metrics["authorized_clients"] = len(authorized_results)
            metrics["unauthorized_clients"] = len(unauthorized_clients)
            
            # Store metrics for history
            self.metrics_history.append({
                "round": server_round,
                "metrics": metrics,
                "num_clients": len(authorized_results),
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            # Record contributions to blockchain
            if self.blockchain and self.current_round_contributions:
                logger.info(f"Recording {len(self.current_round_contributions)} client contributions to blockchain")
                
                for wallet_address, contribution in self.current_round_contributions.items():
                    try:
                        tx_hash = self.blockchain.record_contribution(
                            client_address=wallet_address,
                            round_num=server_round,
                            ipfs_hash=contribution["ipfs_hash"],
                            accuracy=contribution["accuracy"]
                        )
                        logger.info(f"Recorded contribution for {wallet_address}, tx: {tx_hash}")
                    except Exception as e:
                        logger.error(f"Failed to record contribution for {wallet_address}: {e}")
                
                # Allocate rewards for this round
                try:
                    tx_hash = self.blockchain.allocate_rewards_for_round(
                        round_num=server_round,
                        total_reward=self.round_rewards
                    )
                    logger.info(f"Allocated rewards for round {server_round}, tx: {tx_hash}")
                except Exception as e:
                    logger.error(f"Failed to allocate rewards for round {server_round}: {e}")
            
            # Update participating clients in blockchain if available
            if self.blockchain:
                try:
                    # Get the global model hash from the first client's config
                    # Assumes all clients received the same model
                    ipfs_hash = authorized_results[0][1].metrics.get("ipfs_hash", None)
                    
                    if ipfs_hash:
                        # Update model in blockchain with actual client count
                        # Use register_or_update_model instead of register_model
                        tx_hash = self.blockchain.register_or_update_model(
                            ipfs_hash=ipfs_hash,
                            round_num=server_round,
                            version=self.get_version(server_round),
                            participating_clients=len(authorized_results)
                        )
                        logger.info(f"Updated model in blockchain with {len(authorized_results)} clients, tx: {tx_hash}")
                except Exception as e:
                    logger.error(f"Failed to update model in blockchain: {e}")
        
        metrics["rejected_models"] = len(rejected_models)
        return parameters_aggregated, metrics
    
    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: fl.server.client_manager.ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the evaluation round."""
        
        # Get IPFS hash from previous fit round (similar approach to configure_fit)
        weights = parameters_to_ndarrays(parameters)
        
        # Create state dict from weights
        state_dict = {}
        layer_names = ["linear.weight", "linear.bias"]  # Adjust based on your model
        
        for i, name in enumerate(layer_names):
            if i < len(weights):
                state_dict[name] = weights[i].tolist()
        
        # Create metadata with evaluation flag
        model_metadata = {
            "state_dict": state_dict,
            "info": {
                "round": server_round,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "version": self.get_version(server_round),
                "evaluation": True
            }
        }
        
        # Store in IPFS
        ipfs_hash = self.ipfs.add_json(model_metadata)
        logger.info(f"Stored evaluation model in IPFS: {ipfs_hash}")
        
        # Include IPFS hash in config
        config = {"ipfs_hash": ipfs_hash, "server_round": server_round}
        
        # Configure evaluation instructions for clients
        evaluate_ins = EvaluateIns(parameters, config)
        
        # Sample clients for evaluation
        clients = client_manager.sample(
            num_clients=self.min_evaluate_clients, 
            min_num_clients=self.min_available_clients
        )
        
        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]
    
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation results from clients."""
        
        # Filter out unauthorized clients
        authorized_results = []
        unauthorized_clients = []
        
        for client, eval_res in results:
            wallet_address = eval_res.metrics.get("wallet_address", "unknown")
            
            # Check for auth error flag
            if eval_res.metrics.get("error") == "client_not_authorized":
                logger.warning(f"Client {wallet_address} reported as unauthorized")
                unauthorized_clients.append((client, wallet_address))
                continue
            
            # Verify client authorization
            if self.is_client_authorized(wallet_address):
                authorized_results.append((client, eval_res))
            else:
                logger.warning(f"Rejecting evaluation from unauthorized client: {wallet_address}")
                unauthorized_clients.append((client, wallet_address))
        
        if not authorized_results:
            if unauthorized_clients:
                logger.error(f"All {len(unauthorized_clients)} clients were unauthorized. No evaluation aggregation possible.")
            else:
                logger.error("No clients returned evaluation results.")
            return None, {"error": "no_authorized_clients"}
        
        # Perform aggregation with only authorized clients
        loss_aggregated, metrics = super().aggregate_evaluate(server_round, authorized_results, failures)
        
        # Add metrics about client participation
        metrics["total_clients"] = len(results)
        metrics["authorized_clients"] = len(authorized_results)
        metrics["unauthorized_clients"] = len(unauthorized_clients)
        
        # Calculate average accuracy
        accuracies = [res.metrics.get("accuracy", 0.0) for _, res in authorized_results]
        if accuracies:
            avg_accuracy = sum(accuracies) / len(accuracies)
            metrics["avg_accuracy"] = avg_accuracy
        
        # Add evaluation metrics to history
        if loss_aggregated is not None:
            eval_metrics = {
                "round": server_round,
                "eval_loss": loss_aggregated,
                "eval_metrics": metrics,
                "num_clients": len(authorized_results),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            # Save evaluation metrics
            self.metrics_history.append(eval_metrics)
            
            # Log the evaluation results
            logger.info(f"Round {server_round} evaluation: Loss={loss_aggregated:.4f}, Metrics={metrics}")
        
        return loss_aggregated, metrics
    
    def save_metrics_history(self, filepath: str = "metrics/metrics_history.json"):
            """Save metrics history to a file."""
            # Save combined metrics history
            with open(filepath, "w") as f:
                json.dump(self.metrics_history, f, indent=2)
            logger.info(f"Saved metrics history to {filepath}")
            
            # Save individual round metrics to separate files
            metrics_dir = Path(filepath).parent
            for round_metrics in self.metrics_history:
                round_num = round_metrics.get("round", 0)
                round_file = metrics_dir / f"round_{round_num}_metrics.json"
                with open(round_file, "w") as f:
                    json.dump(round_metrics, f, indent=2)
                logger.info(f"Saved round {round_num} metrics to {round_file}")
    
    def save_client_stats(self, filepath: str = "metrics/client_stats.json"):
        """Save client contribution statistics to a file."""
        if not self.blockchain:
            logger.warning("Blockchain connector not available. Cannot save client stats.")
            return
        
        client_stats = {}
        metrics_dir = Path(filepath).parent
        
        try:
            # Get all authorized clients
            clients = self.authorized_clients
            
            for client in clients:
                try:
                    # Get contribution details
                    details = self.blockchain.get_client_contribution_details(client)
                    
                    # Get contribution records
                    records = self.blockchain.get_client_contribution_records(client)
                    
                    # Store in stats
                    client_stats[client] = {
                        "details": details,
                        "records": records
                    }
                    
                    # Save individual client stats to separate files
                    client_file = metrics_dir / f"client_{client[-8:]}_stats.json"
                    with open(client_file, "w") as f:
                        json.dump(client_stats[client], f, indent=2)
                    logger.info(f"Saved client {client[-8:]} stats to {client_file}")
                    
                except Exception as e:
                    logger.error(f"Failed to get stats for client {client}: {e}")
            
            # Save combined stats to file
            with open(filepath, "w") as f:
                json.dump(client_stats, f, indent=2)
                
            logger.info(f"Saved combined client stats to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save client stats: {e}")
            
    def save_model_history(self, filepath: str = "metrics/model_history.json"):
        """Save model history from blockchain to a file."""
        if not self.blockchain:
            logger.warning("Blockchain connector not available. Cannot save model history.")
            return
        
        try:
            # Get model history from blockchain
            models = self.blockchain.get_all_models(self.version_prefix)
            
            metrics_dir = Path(filepath).parent
            
            # Save combined model history
            with open(filepath, "w") as f:
                json.dump(models, f, indent=2)
            
            logger.info(f"Saved model history to {filepath}")
            
            # Save individual model data to separate files
            for model in models:
                round_num = model.get("round", 0)
                model_file = metrics_dir / f"model_round_{round_num}.json"
                
                # Try to get the model data from IPFS
                try:
                    ipfs_hash = model.get("ipfs_hash")
                    if ipfs_hash:
                        model_data = self.ipfs.get_json(ipfs_hash)
                        if model_data:
                            # Save the complete model data including weights
                            with open(model_file, "w") as f:
                                json.dump(model_data, f, indent=2)
                            logger.info(f"Saved round {round_num} model data to {model_file}")
                            
                            # Save a lightweight model info file (without weights)
                            info_file = metrics_dir / f"model_round_{round_num}_info.json"
                            model_info = {**model}
                            if model_data and "info" in model_data:
                                model_info["model_info"] = model_data["info"]
                            with open(info_file, "w") as f:
                                json.dump(model_info, f, indent=2)
                except Exception as e:
                    logger.error(f"Failed to get model data for round {round_num}: {e}")
                    # Save just the model metadata if we couldn't get the full data
                    model_file = metrics_dir / f"model_round_{round_num}_metadata.json"
                    with open(model_file, "w") as f:
                        json.dump(model, f, indent=2)
                    
        except Exception as e:
            logger.error(f"Failed to save model history: {e}")

def start_server(
    server_address: str = "0.0.0.0:8080",
    num_rounds: int = 3,
    min_fit_clients: int = 2,
    min_evaluate_clients: int = 2,
    fraction_fit: float = 1.0,
    ipfs_url: str = "http://127.0.0.1:5001",
    ganache_url: str = "http://127.0.0.1:7545",
    contract_address: Optional[str] = None,
    private_key: Optional[str] = None,
    deploy_contract: bool = False,
    version_prefix: str = "1.0",
    authorized_clients_only: bool = True,
    authorized_clients: Optional[List[str]] = None,
    round_rewards: int = 1000
) -> None:
    """
    Start the enhanced federated learning server with IPFS and blockchain integration.
    
    Args:
        server_address: Server address (host:port)
        num_rounds: Number of federated learning rounds
        min_fit_clients: Minimum number of clients for training
        min_evaluate_clients: Minimum number of clients for evaluation
        fraction_fit: Fraction of clients to use for training
        ipfs_url: IPFS API URL
        ganache_url: Ganache blockchain URL
        contract_address: Federation contract address (if already deployed)
        private_key: Private key for blockchain transactions
        deploy_contract: Whether to deploy a new contract if address not provided
        version_prefix: Version prefix for model versioning
        authorized_clients_only: Whether to only accept contributions from authorized clients
        authorized_clients: List of client addresses to authorize (if not already authorized)
        round_rewards: Reward points to distribute each round
    """
    # Initialize IPFS connector
    ipfs_connector = IPFSConnector(ipfs_api_url=ipfs_url)
    logger.info(f"Initialized IPFS connector: {ipfs_url}")
    
    # Initialize blockchain connector
    blockchain_connector = None
    if ganache_url:
        try:
            blockchain_connector = BlockchainConnector(
                ganache_url=ganache_url,
                contract_address=contract_address,
                private_key=private_key
            )
            
            # Deploy contract if needed
            if contract_address is None and deploy_contract:
                contract_address = blockchain_connector.deploy_contract()
                logger.info(f"Deployed new contract at: {contract_address}")
                
                # Save contract address to file for future use
                with open("contract_address.txt", "w") as f:
                    f.write(contract_address)
                
                # Initialize the blockchain connector with the new contract
                blockchain_connector = BlockchainConnector(
                    ganache_url=ganache_url,
                    contract_address=contract_address,
                    private_key=private_key
                )
            elif contract_address is None:
                logger.warning("No contract address provided and deploy_contract=False. Blockchain features disabled.")
                blockchain_connector = None
                
            # Authorize clients if provided
            if blockchain_connector and authorized_clients:
                # Check which clients are not already authorized
                to_authorize = []
                for client in authorized_clients:
                    if not blockchain_connector.is_client_authorized(client):
                        to_authorize.append(client)
                
                if to_authorize:
                    logger.info(f"Authorizing {len(to_authorize)} new clients")
                    blockchain_connector.authorize_clients(to_authorize)
                else:
                    logger.info("All provided clients are already authorized")
                
        except Exception as e:
            logger.error(f"Failed to initialize blockchain connector: {e}")
            logger.warning("Continuing without blockchain integration")
            blockchain_connector = None
    
    # Configure strategy
    strategy = EnhancedFedAvg(
        fraction_fit=fraction_fit,
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_evaluate_clients,
        min_available_clients=min_fit_clients,
        ipfs_connector=ipfs_connector,
        blockchain_connector=blockchain_connector,
        version_prefix=version_prefix,
        authorized_clients_only=authorized_clients_only,
        round_rewards=round_rewards
    )
    
    # Create metrics directory with timestamp to keep each training run separate
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_dir = Path(f"metrics/run_{timestamp}")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    
    # Start server
    server = fl.server.Server(client_manager=fl.server.SimpleClientManager(), strategy=strategy)
    
    # Run server
    fl.server.start_server(
        server_address=server_address,
        server=server,
        config=fl.server.ServerConfig(num_rounds=num_rounds)
    )
    
    # Save metrics history (both combined and per-round)
    strategy.save_metrics_history(filepath=str(metrics_dir / "metrics_history.json"))
    
    # Save client stats
    strategy.save_client_stats(filepath=str(metrics_dir / "client_stats.json"))
    
    # Save model history
    strategy.save_model_history(filepath=str(metrics_dir / "model_history.json"))
    
    # Create a summary file with key information
    summary = {
        "timestamp": timestamp,
        "num_rounds": num_rounds,
        "min_fit_clients": min_fit_clients,
        "min_evaluate_clients": min_evaluate_clients,
        "authorized_clients_only": authorized_clients_only,
        "version_prefix": version_prefix,
        "contract_address": contract_address,
        "final_metrics": strategy.metrics_history[-1] if strategy.metrics_history else None
    }
    
    with open(metrics_dir / "run_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Server completed {num_rounds} rounds of federated learning")
    logger.info(f"All metrics saved to {metrics_dir}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Start enhanced FL server with IPFS and blockchain integration")
    parser.add_argument("--server-address", type=str, default="0.0.0.0:8080", help="Server address (host:port)")
    parser.add_argument("--rounds", type=int, default=3, help="Number of federated learning rounds")
    parser.add_argument("--min-fit-clients", type=int, default=2, help="Minimum number of clients for training")
    parser.add_argument("--min-evaluate-clients", type=int, default=2, help="Minimum number of clients for evaluation")
    parser.add_argument("--fraction-fit", type=float, default=1.0, help="Fraction of clients to use for training")
    parser.add_argument("--ipfs-url", type=str, default="http://127.0.0.1:5001", help="IPFS API URL")
    parser.add_argument("--ganache-url", type=str, default="http://192.168.1.146:7545", help="Ganache blockchain URL")
    parser.add_argument("--contract-address", type=str, help="Federation contract address")
    parser.add_argument("--private-key", type=str, help="Private key for blockchain transactions")
    parser.add_argument("--deploy-contract", action="store_true", help="Deploy a new contract if address not provided")
    parser.add_argument("--version-prefix", type=str, default="1.0", help="Version prefix for model versioning")
    parser.add_argument("--authorized-clients-only", action="store_true", help="Only accept contributions from authorized clients")
    parser.add_argument("--authorize-clients", nargs="+", help="List of client addresses to authorize")
    parser.add_argument("--round-rewards", type=int, default=1000, help="Reward points to distribute each round")
    
    args = parser.parse_args()
    
    # Check if contract address is stored in file
    if args.contract_address is None and not args.deploy_contract:
        try:
            with open("contract_address.txt", "r") as f:
                args.contract_address = f.read().strip()
                logger.info(f"Loaded contract address from file: {args.contract_address}")
        except FileNotFoundError:
            logger.warning("No contract address provided or found in file")
    
    start_server(
        server_address=args.server_address,
        num_rounds=args.rounds,
        min_fit_clients=args.min_fit_clients,
        min_evaluate_clients=args.min_evaluate_clients,
        fraction_fit=args.fraction_fit,
        ipfs_url=args.ipfs_url,
        ganache_url=args.ganache_url,
        contract_address=args.contract_address,
        private_key=args.private_key,
        deploy_contract=args.deploy_contract,
        version_prefix=args.version_prefix,
        authorized_clients_only=args.authorized_clients_only,
        authorized_clients=args.authorize_clients,
        round_rewards=args.round_rewards
    )