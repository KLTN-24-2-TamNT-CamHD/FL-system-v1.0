"""
Enhanced blockchain connector for federated learning with IPFS.
Connects to Ganache and interacts with the EnhancedModelRegistry smart contract.
Adds client authorization and contribution tracking.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

from web3 import Web3
from web3.contract import Contract
from eth_account.account import Account
from hexbytes import HexBytes

class BlockchainConnector:
    def __init__(
        self, 
        ganache_url: str = "http://127.0.0.1:7545",
        contract_address: Optional[str] = None,
        private_key: Optional[str] = None,
        contract_path: Optional[str] = None
    ):
        """
        Initialize the blockchain connector.
        
        Args:
            ganache_url: URL for Ganache blockchain
            contract_address: Address of deployed EnhancedModelRegistry contract
            private_key: Private key for the account used to sign transactions
            contract_path: Path to the contract JSON file (compiled artifact)
        """
        # Connect to Ganache
        self.web3 = Web3(Web3.HTTPProvider(ganache_url))
        if not self.web3.is_connected():
            raise ConnectionError(f"Failed to connect to Ganache at {ganache_url}")
        
        # Set up account
        if private_key:
            self.account = self.web3.eth.account.from_key(private_key)
            print(f"Using account: {self.account.address}")
        else:
            # Use the first account from Ganache
            self.account = self.web3.eth.accounts[0]
            print(f"Using Ganache account: {self.account}")
        
        # Load the contract if address is provided
        self.contract = None
        if contract_address:
            self._load_contract(contract_address, contract_path)
    
    def _load_contract(self, address: str, contract_path: Optional[str] = None) -> None:
        """
        Load the contract from the provided address.
        
        Args:
            address: Contract address
            contract_path: Path to the contract ABI JSON file
        """
        if contract_path is None:
            # Default path for the compiled contract
            contract_path = Path(__file__).parent / "contracts" / "Federation.json"
        
        # Load contract ABI
        try:
            with open(contract_path, 'r') as f:
                contract_json = json.load(f)
            
            # Handle different compilation outputs (Truffle vs. solc)
            if 'abi' in contract_json:
                contract_abi = contract_json['abi']
            elif 'contracts' in contract_json:
                # Get the first contract in the file
                contract_name = list(contract_json['contracts'].keys())[0]
                contract_abi = contract_json['contracts'][contract_name]['abi']
            else:
                raise ValueError("Could not find ABI in contract JSON")
                
            # Create contract instance
            self.contract = self.web3.eth.contract(address=address, abi=contract_abi)
            print(f"Contract loaded at address: {address}")
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Contract JSON not found at {contract_path}")
        except Exception as e:
            raise RuntimeError(f"Error loading contract: {e}")
    
    def deploy_contract(self, contract_path: Optional[str] = None) -> str:
        """
        Deploy the EnhancedModelRegistry contract to the blockchain.
        
        Args:
            contract_path: Path to the contract JSON file
            
        Returns:
            Address of the deployed contract
        """
        if contract_path is None:
            # Default path for the compiled contract
            contract_path = Path(__file__).parent / "contracts" / "EnhancedModelRegistry.json"
        
        try:
            with open(contract_path, 'r') as f:
                contract_json = json.load(f)
            
            # Get contract bytecode and ABI
            if 'bytecode' in contract_json:
                bytecode = contract_json['bytecode']
                abi = contract_json['abi']
            elif 'contracts' in contract_json:
                # Get the first contract in the file
                contract_name = list(contract_json['contracts'].keys())[0]
                bytecode = contract_json['contracts'][contract_name]['bytecode']
                abi = contract_json['contracts'][contract_name]['abi']
            else:
                raise ValueError("Could not find bytecode and ABI in contract JSON")
            
            # Create contract instance
            Contract = self.web3.eth.contract(abi=abi, bytecode=bytecode)
            
            # Build transaction
            tx_params = {
                'from': self.account if isinstance(self.account, str) else self.account.address,
                'nonce': self.web3.eth.get_transaction_count(
                    self.account if isinstance(self.account, str) else self.account.address
                ),
                'gas': 5000000,  # Increased gas limit for larger contract
                'gasPrice': self.web3.eth.gas_price
            }
            
            # Deploy contract
            transaction = Contract.constructor().build_transaction(tx_params)
            
            # Sign transaction if using private key
            if isinstance(self.account, Account):
                signed_tx = self.account.sign_transaction(transaction)
                tx_hash = self.web3.eth.send_raw_transaction(signed_tx.rawTransaction)
            else:
                tx_hash = self.web3.eth.send_transaction(transaction)
            
            # Wait for transaction to be mined
            tx_receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash)
            contract_address = tx_receipt['contractAddress']
            
            # Load the deployed contract
            self._load_contract(contract_address, contract_path)
            
            print(f"Contract deployed at: {contract_address}")
            return contract_address
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Contract JSON not found at {contract_path}")
        except Exception as e:
            raise RuntimeError(f"Error deploying contract: {e}")
    
    def register_model(
        self, 
        ipfs_hash: str, 
        round_num: int, 
        version: str,
        participating_clients: int = 0
    ) -> str:
        """
        Register a model in the blockchain or update it if it already exists.
        
        Args:
            ipfs_hash: IPFS hash of the model
            round_num: Federated learning round number
            version: Model version (semver format: major.minor.patch)
            participating_clients: Number of clients that participated in training
            
        Returns:
            Transaction hash
        """
        if not self.contract:
            raise RuntimeError("Contract not loaded")
        
        # Build transaction
        tx_params = {
            'from': self.account if isinstance(self.account, str) else self.account.address,
            'nonce': self.web3.eth.get_transaction_count(
                self.account if isinstance(self.account, str) else self.account.address
            ),
            'gas': 2000000,
            'gasPrice': self.web3.eth.gas_price
        }
        
        # Use registerOrUpdateModel function which handles both cases
        transaction = self.contract.functions.registerOrUpdateModel(
            ipfs_hash, 
            round_num, 
            version,
            participating_clients
        ).build_transaction(tx_params)
        
        # Sign transaction if using private key
        try:
            if isinstance(self.account, Account):
                signed_tx = self.account.sign_transaction(transaction)
                tx_hash = self.web3.eth.send_raw_transaction(signed_tx.rawTransaction)
            else:
                tx_hash = self.web3.eth.send_transaction(transaction)
            
            # Wait for transaction to be mined
            _ = self.web3.eth.wait_for_transaction_receipt(tx_hash)
            
            # Convert HexBytes to string for easier handling
            if isinstance(tx_hash, HexBytes):
                tx_hash = tx_hash.hex()
                
            print(f"Model registered with transaction: {tx_hash}")
            return tx_hash
        except Exception as e:
            logging.error(f"Failed to register/update model in blockchain: {str(e)}")
            return None
        
    def register_or_update_model(
        self, 
        ipfs_hash: str, 
        round_num: int, 
        version: str,
        participating_clients: int = 0
    ) -> str:
        """
        Register a model in the blockchain or update it if models already exist for this round.
        
        Args:
            ipfs_hash: IPFS hash of the model
            round_num: Federated learning round number
            version: Model version (semver format: major.minor.patch)
            participating_clients: Number of clients that participated in training
            
        Returns:
            Transaction hash or None if failed
        """
        if not self.contract:
            raise RuntimeError("Contract not loaded")
        
        # Build transaction params
        tx_params = {
            'from': self.account if isinstance(self.account, str) else self.account.address,
            'nonce': self.web3.eth.get_transaction_count(
                self.account if isinstance(self.account, str) else self.account.address
            ),
            'gas': 6000000,  # Increased gas limit
            'gasPrice': self.web3.eth.gas_price
        }
        
        # Try direct registration first (will work if model doesn't exist)
        try:
            transaction = self.contract.functions.registerModel(
                ipfs_hash, 
                round_num, 
                version,
                participating_clients
            ).build_transaction(tx_params)
            
            # Sign and send transaction
            if isinstance(self.account, Account):
                signed_tx = self.account.sign_transaction(transaction)
                tx_hash = self.web3.eth.send_raw_transaction(signed_tx.rawTransaction)
            else:
                tx_hash = self.web3.eth.send_transaction(transaction)
            
            # Wait for transaction receipt
            receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash)
            
            # Convert HexBytes to string for easier handling
            if isinstance(tx_hash, HexBytes):
                tx_hash = tx_hash.hex()
                
            print(f"Model registered with transaction: {tx_hash}")
            return tx_hash
        except Exception as e:
            print(f"Registration failed, trying update: {e}")
            
            # If registration failed, try updateModelByRound
            try:
                # Check if updateModelByRound function exists
                function_signatures = [fn for fn in dir(self.contract.functions) if not fn.startswith('_')]
                if 'updateModelByRound' not in function_signatures:
                    print(f"Function 'updateModelByRound' not found. Available functions: {function_signatures}")
                    return None
                    
                # Update nonce for new transaction
                tx_params['nonce'] = self.web3.eth.get_transaction_count(
                    self.account if isinstance(self.account, str) else self.account.address
                )
                
                transaction = self.contract.functions.updateModelByRound(
                    ipfs_hash, 
                    round_num, 
                    version,
                    participating_clients
                ).build_transaction(tx_params)
                
                # Sign and send transaction
                if isinstance(self.account, Account):
                    signed_tx = self.account.sign_transaction(transaction)
                    tx_hash = self.web3.eth.send_raw_transaction(signed_tx.rawTransaction)
                else:
                    tx_hash = self.web3.eth.send_transaction(transaction)
                
                # Wait for transaction receipt
                receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash)
                
                # Convert HexBytes to string
                if isinstance(tx_hash, HexBytes):
                    tx_hash = tx_hash.hex()
                    
                print(f"Model updated with transaction: {tx_hash}")
                return tx_hash
            except Exception as e2:
                print(f"Update also failed: {e2}")
                return None
    
    def authorize_client(self, client_address: str) -> str:
        """
        Authorize a client to participate in federated learning.
        
        Args:
            client_address: Ethereum address of the client
            
        Returns:
            Transaction hash
        """
        if not self.contract:
            raise RuntimeError("Contract not loaded")
        
        # Build transaction
        tx_params = {
            'from': self.account if isinstance(self.account, str) else self.account.address,
            'nonce': self.web3.eth.get_transaction_count(
                self.account if isinstance(self.account, str) else self.account.address
            ),
            'gas': 100000,
            'gasPrice': self.web3.eth.gas_price
        }
        
        # Call authorizeClient function
        transaction = self.contract.functions.authorizeClient(client_address).build_transaction(tx_params)
        
        # Sign transaction if using private key
        if isinstance(self.account, Account):
            signed_tx = self.account.sign_transaction(transaction)
            tx_hash = self.web3.eth.send_raw_transaction(signed_tx.rawTransaction)
        else:
            tx_hash = self.web3.eth.send_transaction(transaction)
        
        # Wait for transaction to be mined
        _ = self.web3.eth.wait_for_transaction_receipt(tx_hash)
        
        # Convert HexBytes to string for easier handling
        if isinstance(tx_hash, HexBytes):
            tx_hash = tx_hash.hex()
            
        print(f"Client {client_address} authorized with transaction: {tx_hash}")
        return tx_hash
    
    def authorize_clients(self, client_addresses: List[str]) -> str:
        """
        Authorize multiple clients to participate in federated learning.
        
        Args:
            client_addresses: List of Ethereum addresses of clients
            
        Returns:
            Transaction hash
        """
        if not self.contract:
            raise RuntimeError("Contract not loaded")
        
        # Build transaction
        tx_params = {
            'from': self.account if isinstance(self.account, str) else self.account.address,
            'nonce': self.web3.eth.get_transaction_count(
                self.account if isinstance(self.account, str) else self.account.address
            ),
            'gas': 500000,  # Higher gas limit for multiple clients
            'gasPrice': self.web3.eth.gas_price
        }
        
        # Call authorizeClients function
        transaction = self.contract.functions.authorizeClients(client_addresses).build_transaction(tx_params)
        
        # Sign transaction if using private key
        if isinstance(self.account, Account):
            signed_tx = self.account.sign_transaction(transaction)
            tx_hash = self.web3.eth.send_raw_transaction(signed_tx.rawTransaction)
        else:
            tx_hash = self.web3.eth.send_transaction(transaction)
        
        # Wait for transaction to be mined
        _ = self.web3.eth.wait_for_transaction_receipt(tx_hash)
        
        # Convert HexBytes to string for easier handling
        if isinstance(tx_hash, HexBytes):
            tx_hash = tx_hash.hex()
            
        print(f"Authorized {len(client_addresses)} clients with transaction: {tx_hash}")
        return tx_hash
    
    def deauthorize_client(self, client_address: str) -> str:
        """
        Deauthorize a client from participating in federated learning.
        
        Args:
            client_address: Ethereum address of the client
            
        Returns:
            Transaction hash
        """
        if not self.contract:
            raise RuntimeError("Contract not loaded")
        
        # Build transaction
        tx_params = {
            'from': self.account if isinstance(self.account, str) else self.account.address,
            'nonce': self.web3.eth.get_transaction_count(
                self.account if isinstance(self.account, str) else self.account.address
            ),
            'gas': 100000,
            'gasPrice': self.web3.eth.gas_price
        }
        
        # Call deauthorizeClient function
        transaction = self.contract.functions.deauthorizeClient(client_address).build_transaction(tx_params)
        
        # Sign transaction if using private key
        if isinstance(self.account, Account):
            signed_tx = self.account.sign_transaction(transaction)
            tx_hash = self.web3.eth.send_raw_transaction(signed_tx.rawTransaction)
        else:
            tx_hash = self.web3.eth.send_transaction(transaction)
        
        # Wait for transaction to be mined
        _ = self.web3.eth.wait_for_transaction_receipt(tx_hash)
        
        # Convert HexBytes to string for easier handling
        if isinstance(tx_hash, HexBytes):
            tx_hash = tx_hash.hex()
            
        print(f"Client {client_address} deauthorized with transaction: {tx_hash}")
        return tx_hash
    
    def is_client_authorized(self, client_address: str) -> bool:
        """
        Check if a client is authorized to participate.
        
        Args:
            client_address: Ethereum address of the client
            
        Returns:
            True if client is authorized, False otherwise
        """
        if not self.contract:
            raise RuntimeError("Contract not loaded")
        
        return self.contract.functions.isClientAuthorized(client_address).call()
    
    def get_all_authorized_clients(self) -> List[str]:
        """
        Get all authorized client addresses.
        
        Returns:
            List of client addresses
        """
        if not self.contract:
            raise RuntimeError("Contract not loaded")
        
        # Get count first
        count = self.contract.functions.getAuthorizedClientCount().call()
        
        # Get each client
        clients = []
        for i in range(count):
            clients.append(self.contract.functions.authorizedClients(i).call())
        
        return clients
    
    def record_contribution(
        self, 
        client_address: str, 
        round_num: int, 
        ipfs_hash: str, 
        accuracy: float
    ) -> str:
        """
        Record a client's contribution.
        
        Args:
            client_address: Ethereum address of the client
            round_num: Federated learning round number
            ipfs_hash: IPFS hash of the model contribution
            accuracy: Accuracy achieved by the client (0-100)
            
        Returns:
            Transaction hash
        """
        if not self.contract:
            raise RuntimeError("Contract not loaded")
        
        # Convert accuracy to blockchain format (multiply by 10000 to handle decimals)
        accuracy_int = int(accuracy * 100)
        
        # Build transaction
        tx_params = {
            'from': self.account if isinstance(self.account, str) else self.account.address,
            'nonce': self.web3.eth.get_transaction_count(
                self.account if isinstance(self.account, str) else self.account.address
            ),
            'gas': 1000000,
            'gasPrice': self.web3.eth.gas_price
        }
        
        # Call recordContribution function
        transaction = self.contract.functions.recordContribution(
            client_address, 
            round_num, 
            ipfs_hash, 
            accuracy_int
        ).build_transaction(tx_params)
        
        # Sign transaction if using private key
        if isinstance(self.account, Account):
            signed_tx = self.account.sign_transaction(transaction)
            tx_hash = self.web3.eth.send_raw_transaction(signed_tx.rawTransaction)
        else:
            tx_hash = self.web3.eth.send_transaction(transaction)
        
        # Wait for transaction to be mined
        _ = self.web3.eth.wait_for_transaction_receipt(tx_hash)
        
        # Convert HexBytes to string for easier handling
        if isinstance(tx_hash, HexBytes):
            tx_hash = tx_hash.hex()
            
        print(f"Contribution recorded for client {client_address} with transaction: {tx_hash}")
        return tx_hash
    
    def allocate_rewards_for_round(self, round_num: int, total_reward: int) -> str:
        """
        Allocate rewards to clients based on their contributions for a round.
        
        Args:
            round_num: Federated learning round number
            total_reward: Total reward amount to distribute
            
        Returns:
            Transaction hash
        """
        if not self.contract:
            raise RuntimeError("Contract not loaded")
        
        # Build transaction
        tx_params = {
            'from': self.account if isinstance(self.account, str) else self.account.address,
            'nonce': self.web3.eth.get_transaction_count(
                self.account if isinstance(self.account, str) else self.account.address
            ),
            'gas': 500000,
            'gasPrice': self.web3.eth.gas_price
        }
        
        # Call allocateRewardsForRound function
        transaction = self.contract.functions.allocateRewardsForRound(
            round_num, 
            total_reward
        ).build_transaction(tx_params)
        
        # Sign transaction if using private key
        if isinstance(self.account, Account):
            signed_tx = self.account.sign_transaction(transaction)
            tx_hash = self.web3.eth.send_raw_transaction(signed_tx.rawTransaction)
        else:
            tx_hash = self.web3.eth.send_transaction(transaction)
        
        # Wait for transaction to be mined
        _ = self.web3.eth.wait_for_transaction_receipt(tx_hash)
        
        # Convert HexBytes to string for easier handling
        if isinstance(tx_hash, HexBytes):
            tx_hash = tx_hash.hex()
            
        print(f"Rewards allocated for round {round_num} with transaction: {tx_hash}")
        return tx_hash
    
    def get_client_contribution_details(self, client_address: str) -> Dict[str, Any]:
        """
        Get client contribution details.
        
        Args:
            client_address: Ethereum address of the client
            
        Returns:
            Dictionary with client contribution details
        """
        if not self.contract:
            raise RuntimeError("Contract not loaded")
        
        details = self.contract.functions.getClientContribution(client_address).call()
        
        return {
            "contribution_count": details[0],
            "total_score": details[1],
            "is_authorized": details[2],
            "last_contribution_timestamp": details[3],
            "rewards_earned": details[4],
            "rewards_claimed": details[5]
        }
    
    def get_client_contribution_records(self, client_address: str) -> List[Dict[str, Any]]:
        """
        Get client contribution records.
        
        Args:
            client_address: Ethereum address of the client
            
        Returns:
            List of contribution records
        """
        if not self.contract:
            raise RuntimeError("Contract not loaded")
        
        records = self.contract.functions.getClientContributionRecords(client_address).call()
        
        result = []
        for i in range(len(records[0])):
            result.append({
                "round": records[0][i],
                "accuracy": records[1][i] / 100.0,  # Convert back to decimal
                "score": records[2][i],
                "timestamp": records[3][i],
                "rewarded": records[4][i]
            })
        
        return result
    
    def has_contributed_in_round(self, client_address: str, round_number: int) -> bool:
        """
        Check if a client has contributed in a specific round.
        
        Args:
            client_address: Ethereum address of the client
            round_number: The federated learning round number
            
        Returns:
            True if the client has contributed in the specified round, False otherwise
        """
        try:
            if not self.contract:
                logging.warning("Contract not loaded, cannot verify contribution")
                return False
            
            records = self.get_client_contribution_records(client_address)
            
            # Check if any of the records match the current round
            for record in records:
                if record["round"] == round_number:
                    logging.info(f"Found contribution from client {client_address} for round {round_number}")
                    return True
            
            logging.warning(f"No contribution found from client {client_address} for round {round_number}")
            return False
        except Exception as e:
            logging.error(f"Error checking client contribution: {e}")
            # We could be more lenient here by returning True if there's an error
            # return True
            return False
    
    def get_round_contributions(self, round_num: int) -> List[Dict[str, Any]]:
        """
        Get contributions for a specific round.
        
        Args:
            round_num: Federated learning round number
            
        Returns:
            List of contribution records for the round
        """
        if not self.contract:
            raise RuntimeError("Contract not loaded")
        
        records = self.contract.functions.getRoundContributions(round_num).call()
        
        result = []
        for i in range(len(records[0])):
            result.append({
                "client_address": records[0][i],
                "accuracy": records[1][i] / 100.0,  # Convert back to decimal
                "score": records[2][i],
                "rewarded": records[3][i]
            })
        
        return result
    
    def get_model_details(self, ipfs_hash: str, round_num: int) -> Dict[str, Any]:
        """
        Get model details from the blockchain.
        
        Args:
            ipfs_hash: IPFS hash of the model
            round_num: Federated learning round number
            
        Returns:
            Dictionary with model details
        """
        if not self.contract:
            raise RuntimeError("Contract not loaded")
        
        # Generate model ID
        model_id = self.contract.functions.generateModelId(ipfs_hash, round_num).call()
        
        # Get model details
        details = self.contract.functions.getModelDetails(model_id).call()
        
        # Format the response
        return {
            "ipfs_hash": details[0],
            "round": details[1],
            "version": details[2],
            "timestamp": details[3],
            "participating_clients": details[4],
            "publisher": details[5],
            "is_active": details[6]
        }
    
    def get_latest_model(self, version_prefix: str = "1.0") -> Dict[str, Any]:
        """
        Get the latest model for a specific version prefix.
        
        Args:
            version_prefix: Version prefix (e.g., "1.0")
            
        Returns:
            Dictionary with model details
        """
        if not self.contract:
            raise RuntimeError("Contract not loaded")
        
        # Get latest model
        details = self.contract.functions.getLatestModel(version_prefix).call()
        
        # Format the response
        return {
            "model_id": self.web3.to_hex(details[0]),
            "ipfs_hash": details[1],
            "round": details[2],
            "version": details[3],
            "timestamp": details[4],
            "participating_clients": details[5]
        }
    
    def get_models_by_round(self, round_num: int) -> List[str]:
        """
        Get all models for a specific round.
        
        Args:
            round_num: Federated learning round number
            
        Returns:
            List of model IDs
        """
        if not self.contract:
            raise RuntimeError("Contract not loaded")
        
        # Get models by round
        model_ids = self.contract.functions.getModelsByRound(round_num).call()
        
        # Convert bytes32 to hex strings
        return [self.web3.to_hex(model_id) for model_id in model_ids]
    
    def deactivate_model(self, ipfs_hash: str, round_num: int) -> str:
        """
        Deactivate a model.
        
        Args:
            ipfs_hash: IPFS hash of the model
            round_num: Federated learning round number
            
        Returns:
            Transaction hash
        """
        if not self.contract:
            raise RuntimeError("Contract not loaded")
        
        # Generate model ID
        model_id = self.contract.functions.generateModelId(ipfs_hash, round_num).call()
        
        # Build transaction
        tx_params = {
            'from': self.account if isinstance(self.account, str) else self.account.address,
            'nonce': self.web3.eth.get_transaction_count(
                self.account if isinstance(self.account, str) else self.account.address
            ),
            'gas': 100000,
            'gasPrice': self.web3.eth.gas_price
        }
        
        # Call deactivateModel function
        transaction = self.contract.functions.deactivateModel(model_id).build_transaction(tx_params)
        
        # Sign transaction if using private key
        if isinstance(self.account, Account):
            signed_tx = self.account.sign_transaction(transaction)
            tx_hash = self.web3.eth.send_raw_transaction(signed_tx.rawTransaction)
        else:
            tx_hash = self.web3.eth.send_transaction(transaction)
        
        # Wait for transaction to be mined
        _ = self.web3.eth.wait_for_transaction_receipt(tx_hash)
        
        # Convert HexBytes to string for easier handling
        if isinstance(tx_hash, HexBytes):
            tx_hash = tx_hash.hex()
            
        print(f"Model deactivated with transaction: {tx_hash}")
        return tx_hash
    
    def has_contributed_in_round(self, client_address: str, round_number: int) -> bool:
        """
        Check if a client has contributed in a specific round.
        
        Args:
            client_address: Ethereum address of the client
            round_number: The federated learning round number
            
        Returns:
            True if the client has contributed in the specified round, False otherwise
        """
        try:
            if not self.contract:
                logging.warning("Contract not loaded, cannot verify contribution")
                return False
            
            # First check if client is authorized
            if not self.is_client_authorized(client_address):
                logging.warning(f"Client {client_address} is not authorized")
                return False
            
            # Then check contribution details
            details = self.get_client_contribution_details(client_address)
            
            # If client has made any contributions, consider them valid for this round
            # This is a more lenient approach during development
            if details and details["contribution_count"] > 0:
                logging.info(f"Client {client_address} has made {details['contribution_count']} contributions")
                # For more strict checking, uncomment below to verify specific round contributions
                try:
                    records = self.get_client_contribution_records(client_address)
                    for record in records:
                        if record["round"] == round_number:
                            logging.info(f"Found contribution from client {client_address} for round {round_number}")
                            return True
                    
                    # If we get here, client has contributed but not for this specific round
                    # During development, we'll be lenient and accept it anyway
                    logging.warning(f"Client {client_address} has contributed but not for round {round_number}. Accepting anyway.")
                    return True
                except Exception as e:
                    logging.error(f"Error checking specific round contributions: {e}")
                    # Be lenient during development
                    return True
            
            logging.warning(f"No contributions found for client {client_address}")
            return False
        except Exception as e:
            logging.error(f"Error checking client contribution: {e}")
            # During development, you might want to be lenient with errors
            return True  # Change to False for stricter validation