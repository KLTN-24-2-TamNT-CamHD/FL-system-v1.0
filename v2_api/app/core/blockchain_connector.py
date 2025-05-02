# Fixed blockchain_connector.py
import json
import logging
import os
from typing import List, Optional, Dict, Any

from web3 import Web3

logger = logging.getLogger(__name__)

class BlockchainConnector:
    def __init__(self, blockchain_url: str = "http://192.168.1.146:7545", 
                 abi_path: str = None, 
                 contract_address: str = None,
                 private_key: str = None):
        """Initialize the blockchain connector with optional contract address and private key."""
        self.blockchain_url = blockchain_url
        self.web3 = Web3(Web3.HTTPProvider(blockchain_url))
        self.contract_address = contract_address
        self.private_key = private_key
        self.contract = None
        self.abi = None
        
        # Find the ABI file
        if abi_path is None:
            # Try to find the ABI in the assets directory
            current_dir = os.path.dirname(os.path.abspath(__file__))
            possible_paths = [
                os.path.join(current_dir, "assets", "Federation.json"),
                os.path.join(current_dir, "..", "assets", "Federation.json"),
                os.path.join(current_dir, "..", "..", "assets", "Federation.json")
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    self.abi_path = path
                    logger.info(f"ABI path: {self.abi_path}")
                    break
            else:
                logger.error("ABI file not found")
                self.abi_path = None
        else:
            self.abi_path = abi_path
            
        # Load ABI if path is found
        if self.abi_path:
            try:
                with open(self.abi_path, 'r') as file:
                    contract_json = json.load(file)
                    # Handle different formats of ABI files
                    if isinstance(contract_json, dict) and "abi" in contract_json:
                        self.abi = contract_json["abi"]
                    elif isinstance(contract_json, list):
                        # The file might be just the ABI array directly
                        self.abi = contract_json
                    else:
                        logger.error(f"Unexpected ABI format in {self.abi_path}")
                        self.abi = None
            except Exception as e:
                logger.error(f"Failed to load ABI: {str(e)}")
                self.abi = None
        
        # Initialize contract if address is provided
        if self.contract_address and self.abi:
            self.initialize_contract(self.contract_address)
    
    def initialize_contract(self, contract_address: str) -> bool:
        """Initialize the contract with the given address."""
        try:
            if not self.web3.is_connected():
                logger.error("Not connected to blockchain")
                return False
                
            if not self.abi:
                logger.error("ABI not loaded")
                return False
                
            self.contract_address = contract_address
            self.contract = self.web3.eth.contract(
                address=self.web3.to_checksum_address(contract_address),
                abi=self.abi
            )
            logger.info(f"Contract initialized at address: {contract_address}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize contract: {str(e)}")
            return False
    
    def is_connected(self) -> bool:
        """Check if connected to blockchain."""
        return self.web3.is_connected()
    
    def is_contract_initialized(self) -> bool:
        """Check if contract is initialized."""
        return self.contract is not None
    
    def get_authorized_clients(self) -> List[str]:
        """Get list of authorized clients from the contract."""
        if not self.is_contract_initialized():
            logger.error("Contract not initialized")
            return []
            
        try:
            authorized_clients = self.contract.functions.getAuthorizedClients().call()
            return authorized_clients
        except Exception as e:
            logger.error(f"Failed to get authorized clients: {str(e)}")
            return []
    
    def authorize_client(self, client_address: str, account_address: str = None) -> bool:
        """Authorize a client to participate in federated learning."""
        if not self.is_contract_initialized():
            logger.error("Contract not initialized")
            return False
            
        try:
            # Use the provided account or the default account
            if account_address is None and self.private_key:
                account = self.web3.eth.account.from_key(self.private_key)
                account_address = account.address
            
            if not account_address:
                logger.error("No account address provided and no private key set")
                return False
                
            # Build the transaction
            tx = self.contract.functions.authorizeClient(
                self.web3.to_checksum_address(client_address)
            ).build_transaction({
                'from': account_address,
                'nonce': self.web3.eth.get_transaction_count(account_address),
                'gas': 200000,
                'gasPrice': self.web3.eth.gas_price
            })
            
            # Sign and send the transaction
            if self.private_key:
                signed_tx = self.web3.eth.account.sign_transaction(tx, private_key=self.private_key)
                tx_hash = self.web3.eth.send_raw_transaction(signed_tx.rawTransaction)
                receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash)
                return receipt.status == 1
            else:
                # If no private key, assume the node has the account unlocked
                tx_hash = self.web3.eth.send_transaction(tx)
                receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash)
                return receipt.status == 1
                
        except Exception as e:
            logger.error(f"Failed to authorize client: {str(e)}")
            return False
    
    def revoke_client(self, client_address: str, account_address: str = None) -> bool:
        """Revoke a client's authorization."""
        if not self.is_contract_initialized():
            logger.error("Contract not initialized")
            return False
            
        try:
            # Use the provided account or the default account
            if account_address is None and self.private_key:
                account = self.web3.eth.account.from_key(self.private_key)
                account_address = account.address
            
            if not account_address:
                logger.error("No account address provided and no private key set")
                return False
                
            # Build the transaction
            tx = self.contract.functions.revokeClient(
                self.web3.to_checksum_address(client_address)
            ).build_transaction({
                'from': account_address,
                'nonce': self.web3.eth.get_transaction_count(account_address),
                'gas': 200000,
                'gasPrice': self.web3.eth.gas_price
            })
            
            # Sign and send the transaction
            if self.private_key:
                signed_tx = self.web3.eth.account.sign_transaction(tx, private_key=self.private_key)
                tx_hash = self.web3.eth.send_raw_transaction(signed_tx.rawTransaction)
                receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash)
                return receipt.status == 1
            else:
                # If no private key, assume the node has the account unlocked
                tx_hash = self.web3.eth.send_transaction(tx)
                receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash)
                return receipt.status == 1
                
        except Exception as e:
            logger.error(f"Failed to revoke client: {str(e)}")
            return False
    
    def deploy_contract(self, private_key: str = None) -> Optional[str]:
        """Deploy a new contract and return its address."""
        if private_key is None and self.private_key is None:
            logger.error("No private key provided for contract deployment")
            return None
            
        try:
            if not self.web3.is_connected():
                logger.error("Not connected to blockchain")
                return None
                
            if not self.abi:
                logger.error("ABI not loaded")
                return None
                
            # Load the contract bytecode
            with open(self.abi_path, 'r') as file:
                contract_json = json.load(file)
                # Handle different formats of bytecode in ABI files
                if isinstance(contract_json, dict) and "bytecode" in contract_json:
                    bytecode = contract_json["bytecode"]
                elif isinstance(contract_json, list):
                    logger.error("ABI file does not contain bytecode")
                    return None
                else:
                    logger.error("Unexpected ABI format")
                    return None
            
            # Get the contract factory
            contract_factory = self.web3.eth.contract(abi=self.abi, bytecode=bytecode)
            
            # Use the provided private key or the default one
            key_to_use = private_key if private_key else self.private_key
            account = self.web3.eth.account.from_key(key_to_use)
            
            # Build, sign and send the deployment transaction
            transaction = contract_factory.constructor().build_transaction({
                'from': account.address,
                'nonce': self.web3.eth.get_transaction_count(account.address),
                'gas': 2000000,
                'gasPrice': self.web3.eth.gas_price
            })
            
            signed_tx = self.web3.eth.account.sign_transaction(transaction, key_to_use)
            tx_hash = self.web3.eth.send_raw_transaction(signed_tx.rawTransaction)
            tx_receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash)
            
            # Initialize the contract
            self.contract_address = tx_receipt.contractAddress
            self.initialize_contract(self.contract_address)
            
            logger.info(f"Contract deployed at: {self.contract_address}")
            return self.contract_address
            
        except Exception as e:
            logger.error(f"Failed to deploy contract: {str(e)}")
            return None