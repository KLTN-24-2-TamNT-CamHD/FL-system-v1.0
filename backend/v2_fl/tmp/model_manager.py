#!/usr/bin/env python3
"""
Model Manager - Command Line Interface for Federation Smart Contract

This tool provides a command-line interface to interact with the Federation
smart contract for testing model management functionality.

Usage:
    python model_manager.py [command] [arguments]

Commands:
    register_model            - Register a new model
    update_model              - Update an existing model
    get_model_details         - Get details about a specific model
    get_latest_model          - Get the latest model for a specific version
    get_models_by_round       - Get all models for a specific round
    get_latest_model_by_round - Get the latest model for a specific round
    deactivate_model          - Deactivate a model
    register_round_data       - Register both config and model hashes for a round
    verify_round_data         - Verify round data against blockchain record
    get_round_data            - Get round data for a specific round
    help                      - Show this help message
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

# Web3 imports
from web3 import Web3, HTTPProvider
from web3.exceptions import ContractLogicError
from eth_account import Account
from eth_account.signers.local import LocalAccount

# IPFS client implementation
import ipfshttpclient

# Set up logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ModelManager")


class ModelManager:
    """Interface to the Federation smart contract for model management."""

    def __init__(self, config_path: str = "blockchain_config.json"):
        """Initialize the ModelManager with the contract instance."""
        self.config = self._load_config(config_path)
        self.w3 = self._initialize_web3()
        self.account = self._initialize_account()
        self.contract = self._initialize_contract()
        self.ipfs = self._initialize_ipfs()

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from a JSON file."""
        try:
            with open(config_path, 'r') as file:
                config = json.load(file)
                logger.info(f"Configuration loaded from {config_path}")
                return config
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {config_path}")
            raise
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in configuration file: {config_path}")
            raise

    def _initialize_web3(self) -> Web3:
        """Initialize Web3 connection."""
        provider_url = self.config.get("provider_url", "http://127.0.0.1:8545")
        w3 = Web3(HTTPProvider(provider_url))
        
        if not w3.is_connected():
            logger.error(f"Could not connect to Ethereum node at {provider_url}")
            raise ConnectionError(f"Failed to connect to Ethereum node at {provider_url}")
        
        logger.info(f"Connected to Ethereum node at {provider_url}")
        return w3

    def _initialize_account(self) -> LocalAccount:
        """Initialize Ethereum account from private key."""
        private_key = self.config.get("private_key")
        if not private_key:
            logger.error("Private key not found in configuration")
            raise ValueError("Private key is required in configuration")
        
        account = Account.from_key(private_key)
        logger.info(f"Account initialized: {account.address}")
        return account

    def _initialize_contract(self) -> Any:
        """Initialize the contract instance."""
        contract_address = self.config.get("contract_address")
        contract_abi_path = self.config.get("contract_abi_path")
        
        if not contract_address:
            logger.error("Contract address not found in configuration")
            raise ValueError("Contract address is required in configuration")
        
        if not contract_abi_path:
            logger.error("Contract ABI path not found in configuration")
            raise ValueError("Contract ABI path is required in configuration")
        
        try:
            with open(contract_abi_path, 'r') as file:
                contract_abi = json.load(file)
        except FileNotFoundError:
            logger.error(f"Contract ABI file not found: {contract_abi_path}")
            raise
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in contract ABI file: {contract_abi_path}")
            raise
        
        contract = self.w3.eth.contract(address=contract_address, abi=contract_abi)
        logger.info(f"Contract initialized at address {contract_address}")
        return contract

    def _initialize_ipfs(self) -> ipfshttpclient.client.Client:
        """Initialize IPFS client."""
        ipfs_api = self.config.get("ipfs_api", "/ip4/127.0.0.1/tcp/5001")
        try:
            client = ipfshttpclient.connect(ipfs_api)
            logger.info(f"Connected to IPFS node at {ipfs_api}")
            return client
        except Exception as e:
            logger.error(f"Failed to connect to IPFS node at {ipfs_api}: {e}")
            logger.warning("IPFS functionality will be limited")
            return None

    def _build_transaction(self, function_call):
        """Build a transaction for a contract function call."""
        return function_call.build_transaction({
            'from': self.account.address,
            'nonce': self.w3.eth.get_transaction_count(self.account.address),
            'gas': 2000000,
            'gasPrice': self.w3.eth.gas_price
        })

    def _sign_and_send_transaction(self, transaction):
        """Sign and send a transaction."""
        signed_tx = self.account.sign_transaction(transaction)
        tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
        return tx_hash

    def _wait_for_transaction(self, tx_hash):
        """Wait for a transaction to be mined."""
        logger.info(f"Waiting for transaction {tx_hash.hex()} to be mined...")
        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
        logger.info(f"Transaction mined. Status: {'Success' if receipt.status == 1 else 'Failed'}")
        return receipt

    def _ipfs_add_json(self, data: Dict[str, Any]) -> str:
        """Add JSON data to IPFS and return the hash."""
        if not self.ipfs:
            logger.error("IPFS client not initialized")
            return "ipfs-hash-placeholder"
        
        try:
            json_str = json.dumps(data)
            res = self.ipfs.add_json(json_str)
            logger.info(f"Added JSON to IPFS with hash: {res}")
            return res
        except Exception as e:
            logger.error(f"Failed to add JSON to IPFS: {e}")
            return "ipfs-hash-error"

    def _ipfs_get_json(self, ipfs_hash: str) -> Dict[str, Any]:
        """Get JSON data from IPFS."""
        if not self.ipfs:
            logger.error("IPFS client not initialized")
            return {}
        
        try:
            data = self.ipfs.get_json(ipfs_hash)
            logger.info(f"Retrieved JSON from IPFS with hash: {ipfs_hash}")
            return data
        except Exception as e:
            logger.error(f"Failed to get JSON from IPFS: {e}")
            return {}

    # Model management functions
    def register_model(self, ipfs_hash: str, round_num: int, version: str, participating_clients: int = 0) -> str:
        """Register a new model in the contract."""
        try:
            # Call the contract function
            function_call = self.contract.functions.registerModel(
                ipfs_hash, 
                round_num, 
                version, 
                participating_clients
            )
            
            # Build, sign, and send the transaction
            tx = self._build_transaction(function_call)
            tx_hash = self._sign_and_send_transaction(tx)
            receipt = self._wait_for_transaction(tx_hash)
            
            if receipt.status == 1:
                # Calculate the model ID
                model_id = self.contract.functions.generateModelId(ipfs_hash, round_num).call()
                logger.info(f"Model registered successfully with ID: {model_id.hex()}")
                return model_id.hex()
            else:
                logger.error("Model registration failed")
                return ""
                
        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            return ""

    def update_model(self, ipfs_hash: str, round_num: int, version: str, participating_clients: int = 0) -> str:
        """Update an existing model by round."""
        try:
            # Call the contract function
            function_call = self.contract.functions.updateModelByRound(
                ipfs_hash, 
                round_num, 
                version, 
                participating_clients
            )
            
            # Build, sign, and send the transaction
            tx = self._build_transaction(function_call)
            tx_hash = self._sign_and_send_transaction(tx)
            receipt = self._wait_for_transaction(tx_hash)
            
            if receipt.status == 1:
                # Calculate the model ID
                model_id = self.contract.functions.generateModelId(ipfs_hash, round_num).call()
                logger.info(f"Model updated successfully with ID: {model_id.hex()}")
                return model_id.hex()
            else:
                logger.error("Model update failed")
                return ""
                
        except Exception as e:
            logger.error(f"Failed to update model: {e}")
            return ""

    def get_model_details(self, model_id: str) -> Dict[str, Any]:
        """Get details about a specific model."""
        try:
            # Convert the hex string to bytes32
            model_id_bytes = bytes.fromhex(model_id.replace("0x", ""))
            
            # Call the contract function
            result = self.contract.functions.getModelDetails(model_id_bytes).call()
            
            # Format the result
            model_details = {
                "ipfs_hash": result[0],
                "round": result[1],
                "version": result[2],
                "timestamp": datetime.utcfromtimestamp(result[3]).isoformat(),
                "participating_clients": result[4],
                "publisher": result[5],
                "is_active": result[6]
            }
            
            logger.info(f"Retrieved model details for ID: {model_id}")
            return model_details
                
        except ContractLogicError as e:
            logger.error(f"Contract error: {e}")
            return {"error": str(e)}
        except Exception as e:
            logger.error(f"Failed to get model details: {e}")
            return {"error": str(e)}

    def get_latest_model(self, version_prefix: str) -> Dict[str, Any]:
        """Get the latest model for a specific version prefix."""
        try:
            # Call the contract function
            result = self.contract.functions.getLatestModel(version_prefix).call()
            
            # Format the result
            model_details = {
                "model_id": result[0].hex(),
                "ipfs_hash": result[1],
                "round": result[2],
                "version": result[3],
                "timestamp": datetime.utcfromtimestamp(result[4]).isoformat(),
                "participating_clients": result[5]
            }
            
            logger.info(f"Retrieved latest model for version prefix: {version_prefix}")
            return model_details
                
        except ContractLogicError as e:
            logger.error(f"Contract error: {e}")
            return {"error": str(e)}
        except Exception as e:
            logger.error(f"Failed to get latest model: {e}")
            return {"error": str(e)}

    def get_models_by_round(self, round_num: int) -> List[str]:
        """Get all models for a specific round."""
        try:
            # Call the contract function
            result = self.contract.functions.getModelsByRound(round_num).call()
            
            # Convert bytes32 to hex strings
            model_ids = [model_id.hex() for model_id in result]
            
            logger.info(f"Retrieved {len(model_ids)} models for round: {round_num}")
            return model_ids
                
        except Exception as e:
            logger.error(f"Failed to get models by round: {e}")
            return []

    def get_latest_model_by_round(self, round_num: int) -> Dict[str, Any]:
        """Get the latest model for a specific round."""
        try:
            # Call the contract function
            result = self.contract.functions.getLatestModelByRound(round_num).call()
            
            # Format the result
            model_details = {
                "model_id": result[0].hex(),
                "ipfs_hash": result[1],
                "version": result[2],
                "timestamp": datetime.utcfromtimestamp(result[3]).isoformat(),
                "participating_clients": result[4],
                "publisher": result[5],
                "is_active": result[6]
            }
            
            logger.info(f"Retrieved latest model for round: {round_num}")
            return model_details
                
        except ContractLogicError as e:
            logger.error(f"Contract error: {e}")
            return {"error": str(e)}
        except Exception as e:
            logger.error(f"Failed to get latest model by round: {e}")
            return {"error": str(e)}

    def deactivate_model(self, model_id: str) -> bool:
        """Deactivate a model."""
        try:
            # Convert the hex string to bytes32
            model_id_bytes = bytes.fromhex(model_id.replace("0x", ""))
            
            # Call the contract function
            function_call = self.contract.functions.deactivateModel(model_id_bytes)
            
            # Build, sign, and send the transaction
            tx = self._build_transaction(function_call)
            tx_hash = self._sign_and_send_transaction(tx)
            receipt = self._wait_for_transaction(tx_hash)
            
            if receipt.status == 1:
                logger.info(f"Model deactivated successfully: {model_id}")
                return True
            else:
                logger.error(f"Failed to deactivate model: {model_id}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to deactivate model: {e}")
            return False

    # RoundData management functions (if implemented in contract)
    def register_round_data(self, config_ipfs_hash: str, model_ipfs_hash: str, 
                           round_num: int, version: str) -> str:
        """Register both config and model hashes for a round."""
        try:
            # Check if the contract has the registerRoundData function
            if not hasattr(self.contract.functions, 'registerRoundData'):
                logger.error("Contract does not have registerRoundData function")
                return ""
            
            # Call the contract function
            function_call = self.contract.functions.registerRoundData(
                config_ipfs_hash, 
                model_ipfs_hash, 
                round_num, 
                version
            )
            
            # Build, sign, and send the transaction
            tx = self._build_transaction(function_call)
            tx_hash = self._sign_and_send_transaction(tx)
            receipt = self._wait_for_transaction(tx_hash)
            
            if receipt.status == 1:
                # The function might return a round data ID
                logger.info(f"Round data registered successfully for round: {round_num}")
                return tx_hash.hex()
            else:
                logger.error("Round data registration failed")
                return ""
                
        except Exception as e:
            logger.error(f"Failed to register round data: {e}")
            return ""

    def verify_round_data(self, config_ipfs_hash: str, model_ipfs_hash: str, 
                         round_num: int, tx_hash: str) -> bool:
        """Verify round data against blockchain record."""
        try:
            # Check if the contract has the verifyRoundData function
            if not hasattr(self.contract.functions, 'verifyRoundData'):
                logger.error("Contract does not have verifyRoundData function")
                return False
            
            # Convert the hex string to bytes32
            tx_hash_bytes = bytes.fromhex(tx_hash.replace("0x", ""))
            
            # Call the contract function
            result = self.contract.functions.verifyRoundData(
                config_ipfs_hash, 
                model_ipfs_hash, 
                round_num, 
                tx_hash_bytes
            ).call()
            
            logger.info(f"Round data verification result: {result}")
            return result
                
        except Exception as e:
            logger.error(f"Failed to verify round data: {e}")
            return False

    def get_round_data(self, round_num: int) -> Dict[str, Any]:
        """Get round data for a specific round."""
        try:
            # Check if the contract has the getRoundData function
            if not hasattr(self.contract.functions, 'getRoundData'):
                logger.error("Contract does not have getRoundData function")
                return {"error": "Function not implemented"}
            
            # Call the contract function
            result = self.contract.functions.getRoundData(round_num).call()
            
            # Format the result
            round_data = {
                "config_ipfs_hash": result[0],
                "model_ipfs_hash": result[1],
                "version": result[2],
                "timestamp": datetime.utcfromtimestamp(result[3]).isoformat(),
                "tx_hash": result[4].hex(),
                "participating_clients": result[5],
                "is_active": result[6]
            }
            
            logger.info(f"Retrieved round data for round: {round_num}")
            return round_data
                
        except ContractLogicError as e:
            logger.error(f"Contract error: {e}")
            return {"error": str(e)}
        except Exception as e:
            logger.error(f"Failed to get round data: {e}")
            return {"error": str(e)}

    # IPFS utility functions
    def create_mock_model_data(self, round_num: int, is_ensemble: bool = False) -> Dict[str, Any]:
        """Create mock model data for testing."""
        if is_ensemble:
            # Create mock ensemble model
            ensemble_state = {
                "model_names": ["linear", "forest", "svm"],
                "weights": [0.4, 0.3, 0.3],
                "ensemble_type": "weighted_average"
            }
            
            model_data = {
                "ensemble_state": ensemble_state,
                "info": {
                    "round": round_num,
                    "timestamp": datetime.now().isoformat(),
                    "is_ensemble": True,
                    "num_models": len(ensemble_state["model_names"]),
                    "model_names": ensemble_state["model_names"]
                }
            }
        else:
            # Create mock regular model
            state_dict = {
                "linear.weight": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
                "linear.bias": [0.01, 0.02]
            }
            
            model_data = {
                "state_dict": state_dict,
                "info": {
                    "round": round_num,
                    "timestamp": datetime.now().isoformat(),
                    "is_ensemble": False
                }
            }
        
        return model_data

    def create_mock_config_data(self, round_num: int) -> Dict[str, Any]:
        """Create mock configuration data for testing."""
        config_data = {
            "config": {
                "server_round": round_num,
                "ga_stacking": True,
                "local_epochs": 5,
                "validation_split": 0.2,
                "is_ensemble": True,
                "training_params": {
                    "batch_size": 32,
                    "learning_rate": 0.01
                }
            },
            "info": {
                "round": round_num,
                "timestamp": datetime.now().isoformat(),
                "version": f"1.0.{round_num}"
            }
        }
        
        return config_data

    def upload_mock_data_to_ipfs(self, round_num: int) -> Tuple[str, str]:
        """Upload mock model and config data to IPFS for testing."""
        # Create and upload mock model data
        model_data = self.create_mock_model_data(round_num, is_ensemble=True)
        model_ipfs_hash = self._ipfs_add_json(model_data)
        
        # Create and upload mock config data
        config_data = self.create_mock_config_data(round_num)
        config_ipfs_hash = self._ipfs_add_json(config_data)
        
        return config_ipfs_hash, model_ipfs_hash


# Command-line interface
def main():
    parser = argparse.ArgumentParser(description="Model Manager CLI for Federation Smart Contract")
    parser.add_argument("command", help="Command to execute", 
                      choices=["register_model", "update_model", "get_model_details", 
                               "get_latest_model", "get_models_by_round", 
                               "get_latest_model_by_round", "deactivate_model",
                               "register_round_data", "verify_round_data", "get_round_data",
                               "upload_mock_data", "help"])
    
    # Common arguments
    parser.add_argument("--config", "-c", help="Path to blockchain config file", 
                      default="blockchain_config.json")
    
    # Command-specific arguments
    parser.add_argument("--ipfs-hash", help="IPFS hash for the model or config")
    parser.add_argument("--config-hash", help="IPFS hash for the config")
    parser.add_argument("--model-hash", help="IPFS hash for the model")
    parser.add_argument("--round", "-r", type=int, help="Training round number")
    parser.add_argument("--version", "-v", help="Model version")
    parser.add_argument("--clients", type=int, help="Number of participating clients", default=0)
    parser.add_argument("--model-id", help="Model ID (bytes32 as hex string)")
    parser.add_argument("--tx-hash", help="Transaction hash for verification")
    parser.add_argument("--use-mock", action="store_true", help="Use mock data (creates and uploads to IPFS)")
    
    args = parser.parse_args()
    
    # Initialize model manager
    try:
        manager = ModelManager(args.config)
    except Exception as e:
        logger.error(f"Failed to initialize model manager: {e}")
        sys.exit(1)
    
    # Execute the requested command
    if args.command == "help":
        parser.print_help()
        return
    
    # Handle the use_mock flag for all commands
    if args.use_mock and not args.round:
        logger.error("Round number is required when using mock data")
        sys.exit(1)
    
    if args.use_mock:
        logger.info(f"Using mock data for round {args.round}")
        config_hash, model_hash = manager.upload_mock_data_to_ipfs(args.round)
        logger.info(f"Uploaded mock data to IPFS: config_hash={config_hash}, model_hash={model_hash}")
        
        # Set the hashes for the command if they weren't provided
        if not args.ipfs_hash:
            args.ipfs_hash = model_hash
        if not args.config_hash:
            args.config_hash = config_hash
        if not args.model_hash:
            args.model_hash = model_hash
    
    # Execute the command
    try:
        if args.command == "register_model":
            if not all([args.ipfs_hash, args.round is not None, args.version]):
                logger.error("Missing required arguments: --ipfs-hash, --round, --version")
                sys.exit(1)
            
            result = manager.register_model(args.ipfs_hash, args.round, args.version, args.clients)
            print(json.dumps({"model_id": result}, indent=2))
        
        elif args.command == "update_model":
            if not all([args.ipfs_hash, args.round is not None, args.version]):
                logger.error("Missing required arguments: --ipfs-hash, --round, --version")
                sys.exit(1)
            
            result = manager.update_model(args.ipfs_hash, args.round, args.version, args.clients)
            print(json.dumps({"model_id": result}, indent=2))
        
        elif args.command == "get_model_details":
            if not args.model_id:
                logger.error("Missing required argument: --model-id")
                sys.exit(1)
            
            result = manager.get_model_details(args.model_id)
            print(json.dumps(result, indent=2))
        
        elif args.command == "get_latest_model":
            if not args.version:
                logger.error("Missing required argument: --version")
                sys.exit(1)
            
            result = manager.get_latest_model(args.version)
            print(json.dumps(result, indent=2))
        
        elif args.command == "get_models_by_round":
            if args.round is None:
                logger.error("Missing required argument: --round")
                sys.exit(1)
            
            result = manager.get_models_by_round(args.round)
            print(json.dumps({"model_ids": result}, indent=2))
        
        elif args.command == "get_latest_model_by_round":
            if args.round is None:
                logger.error("Missing required argument: --round")
                sys.exit(1)
            
            result = manager.get_latest_model_by_round(args.round)
            print(json.dumps(result, indent=2))
        
        elif args.command == "deactivate_model":
            if not args.model_id:
                logger.error("Missing required argument: --model-id")
                sys.exit(1)
            
            result = manager.deactivate_model(args.model_id)
            print(json.dumps({"success": result}, indent=2))
        
        elif args.command == "register_round_data":
            if not all([args.config_hash, args.model_hash, args.round is not None, args.version]):
                logger.error("Missing required arguments: --config-hash, --model-hash, --round, --version")
                sys.exit(1)
            
            result = manager.register_round_data(args.config_hash, args.model_hash, args.round, args.version)
            print(json.dumps({"tx_hash": result}, indent=2))
        
        elif args.command == "verify_round_data":
            if not all([args.config_hash, args.model_hash, args.round is not None, args.tx_hash]):
                logger.error("Missing required arguments: --config-hash, --model-hash, --round, --tx-hash")
                sys.exit(1)
            
            result = manager.verify_round_data(args.config_hash, args.model_hash, args.round, args.tx_hash)
            print(json.dumps({"verified": result}, indent=2))
        
        elif args.command == "get_round_data":
            if args.round is None:
                logger.error("Missing required argument: --round")
                sys.exit(1)
            
            result = manager.get_round_data(args.round)
            print(json.dumps(result, indent=2))
        
        elif args.command == "upload_mock_data":
            if args.round is None:
                logger.error("Missing required argument: --round")
                sys.exit(1)
            
            config_hash, model_hash = manager.upload_mock_data_to_ipfs(args.round)
            print(json.dumps({
                "config_hash": config_hash,
                "model_hash": model_hash
            }, indent=2))
    
    except Exception as e:
        logger.error(f"Command execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
